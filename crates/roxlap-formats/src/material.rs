//! Voxel materials — opacity + blend mode for transparent voxels (TV stage).
//!
//! Every voxel roxlap renders is opaque today: both backends are per-pixel
//! front-to-back 3D-DDA raymarchers that stop at the first solid voxel. A
//! *material* attaches an opacity and a blend mode to a voxel so the march
//! can instead accumulate it front-to-back over what lies behind (the
//! per-pixel DDA visits cells strictly front-to-back, so this is
//! order-correct without any sorting — see `PORTING-TRANSPARENCY.md`).
//!
//! Materials are kept out of the `0x80RRGGBB` colour word: its high byte is
//! voxlap's lightmode-1 *brightness*, not alpha (see [`crate::kv6`] and
//! [`crate::vxl`]). Instead a voxel carries a one-byte **material id** that
//! indexes a [`MaterialTable`] — a 256-entry global palette the renderer
//! owns. Id `0` is permanently [`Material::OPAQUE`], so a model or grid that
//! carries no material data resolves every voxel to id 0 and renders exactly
//! as before.

/// How a voxel's colour combines with what is already behind it along a ray.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Fully opaque: the first solid hit wins and occludes everything behind
    /// it — the existing render path. A voxel's `alpha` is ignored.
    #[default]
    Opaque = 0,
    /// Front-to-back `over` compositing; `alpha` is the voxel's opacity.
    /// Glass, smoke, water.
    AlphaBlend = 1,
    /// Commutative additive glow: contributes `alpha`·colour to the pixel
    /// without occluding what is behind it (order-independent). Spells,
    /// fire, magic auras, muzzle flashes.
    Additive = 2,
}

impl BlendMode {
    /// Decode the on-wire `u8`. Returns `None` for an unknown discriminant.
    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Opaque),
            1 => Some(Self::AlphaBlend),
            2 => Some(Self::Additive),
            _ => None,
        }
    }

    /// The on-wire discriminant.
    #[must_use]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// One material: an opacity and a blend mode, indexed out of a
/// [`MaterialTable`] by a per-voxel material id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Material {
    /// Opacity for [`BlendMode::AlphaBlend`] / intensity scale for
    /// [`BlendMode::Additive`]; ignored for [`BlendMode::Opaque`].
    /// `0` = fully transparent, `255` = fully opaque / full intensity.
    pub alpha: u8,
    /// How the voxel composites with what is behind it.
    pub mode: BlendMode,
}

impl Material {
    /// The reserved, fully-opaque material at table id `0` — the
    /// back-compat default for every voxel without explicit material data.
    pub const OPAQUE: Self = Self {
        alpha: 255,
        mode: BlendMode::Opaque,
    };

    /// An [`BlendMode::AlphaBlend`] material with opacity `alpha`.
    #[must_use]
    pub fn alpha_blend(alpha: u8) -> Self {
        Self {
            alpha,
            mode: BlendMode::AlphaBlend,
        }
    }

    /// An [`BlendMode::Additive`] glow material scaled by `alpha`.
    #[must_use]
    pub fn additive(alpha: u8) -> Self {
        Self {
            alpha,
            mode: BlendMode::Additive,
        }
    }

    /// True for [`BlendMode::Opaque`] — the first-hit, fully-occluding path.
    #[must_use]
    pub fn is_opaque(self) -> bool {
        matches!(self.mode, BlendMode::Opaque)
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::OPAQUE
    }
}

/// A 256-entry palette of [`Material`]s indexed by a per-voxel `u8` material
/// id. The renderer owns one table (a global palette); voxels reference
/// materials by id rather than embedding them.
///
/// Id `0` is permanently [`Material::OPAQUE`] and cannot be redefined —
/// it is the value every material-free voxel resolves to, so the opaque
/// world stays byte-for-byte unchanged. [`set`](Self::set) silently
/// ignores id 0.
#[derive(Clone, Debug)]
pub struct MaterialTable {
    materials: [Material; 256],
}

impl MaterialTable {
    /// A fresh palette: every id is [`Material::OPAQUE`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            materials: [Material::OPAQUE; 256],
        }
    }

    /// Define material `id`. Id `0` is reserved as [`Material::OPAQUE`] and
    /// cannot be overwritten — defining it is a no-op that returns `false`;
    /// any other id returns `true`.
    pub fn set(&mut self, id: u8, mat: Material) -> bool {
        if id == 0 {
            return false;
        }
        self.materials[id as usize] = mat;
        true
    }

    /// The material at `id` ([`Material::OPAQUE`] for any never-set id).
    #[must_use]
    pub fn get(&self, id: u8) -> Material {
        self.materials[id as usize]
    }

    /// True when every id is [`BlendMode::Opaque`] — lets a backend skip the
    /// whole transparency path while nothing translucent is defined.
    #[must_use]
    pub fn all_opaque(&self) -> bool {
        self.materials.iter().all(|m| m.is_opaque())
    }

    /// The backing 256-entry array, for backends that upload it wholesale.
    #[must_use]
    pub fn as_array(&self) -> &[Material; 256] {
        &self.materials
    }
}

impl Default for MaterialTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_mode_round_trips() {
        for m in [
            BlendMode::Opaque,
            BlendMode::AlphaBlend,
            BlendMode::Additive,
        ] {
            assert_eq!(BlendMode::from_u8(m.as_u8()), Some(m));
        }
        assert_eq!(BlendMode::from_u8(3), None);
        assert_eq!(BlendMode::default(), BlendMode::Opaque);
    }

    #[test]
    fn material_defaults_opaque() {
        assert_eq!(Material::default(), Material::OPAQUE);
        assert!(Material::OPAQUE.is_opaque());
        assert!(!Material::alpha_blend(128).is_opaque());
        assert!(!Material::additive(200).is_opaque());
    }

    #[test]
    fn table_starts_all_opaque() {
        let t = MaterialTable::new();
        assert!(t.all_opaque());
        for id in 0..=255u8 {
            assert_eq!(t.get(id), Material::OPAQUE);
        }
    }

    #[test]
    fn table_set_and_get() {
        let mut t = MaterialTable::new();
        assert!(t.set(1, Material::alpha_blend(64)));
        assert_eq!(t.get(1), Material::alpha_blend(64));
        assert!(!t.all_opaque());
        assert!(t.set(255, Material::additive(255)));
        assert_eq!(t.get(255), Material::additive(255));
    }

    #[test]
    fn id_zero_is_locked_opaque() {
        let mut t = MaterialTable::new();
        assert!(!t.set(0, Material::alpha_blend(0)));
        assert_eq!(t.get(0), Material::OPAQUE);
        assert!(t.all_opaque());
    }
}

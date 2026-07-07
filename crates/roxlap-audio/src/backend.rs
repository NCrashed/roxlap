//! AU.2 — the playback boundary: the [`AudioOut`] trait the acoustics
//! core feeds, and the pure [`SourcePool`] slot policy shared by any
//! backend.
//!
//! Every backend type stays behind this trait (hazard 1 of the entry
//! doc: kira has a history of breaking API redesigns — the acoustics
//! core and the hosts must compile without it). The kira
//! implementation lives in [`crate::kira_out`] behind the `kira`
//! cargo feature.

use glam::{DQuat, DVec3};

use crate::{synth::SoundBuffer, ListenerAcoustics, SourceAcoustics};

/// A registered sound asset (backend-owned decoded buffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoundKey(pub(crate) usize);

/// One live spatial source (a pooled backend track). Stale after the
/// source is stopped or stolen — backend calls on a stale id are
/// no-ops by contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceId(pub(crate) u64);

/// The playback backend the acoustics core drives. Implementations
/// apply parameter changes with **tweens** (~120 ms for per-source
/// occlusion, ~1 s for the listener's reverb) — never as raw jumps
/// (hazard 3: parameter zippering).
pub trait AudioOut {
    /// Register a mono PCM buffer; the key is valid for this backend's
    /// lifetime.
    fn register(&mut self, sound: &SoundBuffer) -> SoundKey;

    /// Update the listener pose (world position + orientation).
    fn set_listener(&mut self, pos: DVec3, orientation: DQuat);

    /// Fire a one-shot at a world position. `initial` sets the source's
    /// occlusion parameters **instantly** at spawn (no tween) — a fresh
    /// voice has nothing to zipper against, and a one-shot's envelope is
    /// often gone before a 120 ms ramp would arrive, so it must start
    /// already muffled; pass `None` for a fully-clear start. Returns
    /// `None` when every pooled track is busy with something more
    /// important (the pool steals finished/oldest one-shots first, never
    /// loops).
    fn play(
        &mut self,
        sound: SoundKey,
        at: DVec3,
        initial: Option<&SourceAcoustics>,
    ) -> Option<SourceId>;

    /// Start a looping source (crystal hum) with the same instant
    /// `initial` occlusion as [`play`](Self::play) — a hum entering
    /// earshot through rock must start muffled, not blip clear for
    /// 120 ms. Loops hold their track until [`stop`](Self::stop).
    fn play_loop(
        &mut self,
        sound: SoundKey,
        at: DVec3,
        initial: Option<&SourceAcoustics>,
    ) -> Option<SourceId>;

    /// Stop a source and free its track. Stale ids are ignored.
    fn stop(&mut self, id: SourceId);

    /// Move a live source (tracked emitters). Stale ids are ignored.
    fn set_source_position(&mut self, id: SourceId, pos: DVec3);

    /// Apply occlusion parameters to a live source (~120 ms tween).
    fn apply_source(&mut self, id: SourceId, acoustics: &SourceAcoustics);

    /// Apply the listener's reverb environment (~1 s tween).
    fn apply_listener(&mut self, acoustics: &ListenerAcoustics);
}

/// How a pooled slot is currently used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotUse {
    Free,
    /// A one-shot: stealable once finished, and the OLDEST one is
    /// stolen when the pool is full (a new sound beats a stale tail).
    OneShot,
    /// A loop: never stolen; freed only by an explicit stop.
    Looping,
}

/// Backend-agnostic voice-slot policy for a fixed pool of spatial
/// tracks — pure and unit-tested here so every backend (the built-in
/// kira one, or a host's own [`AudioOut`]) inherits the same behaviour
/// (hazard 2: tracks are created up front and reused; a per-shot
/// allocation storm is the classic mistake). One-shots are stealable
/// (oldest first) when the pool is full; loops hold their slot until
/// explicitly stopped.
#[derive(Debug)]
pub struct SourcePool {
    slots: Vec<Slot>,
    next_id: u64,
    clock: u64,
}

#[derive(Debug, Clone, Copy)]
struct Slot {
    used: SlotUse,
    id: SourceId,
    started: u64,
}

impl SourcePool {
    /// A pool of `capacity` voice slots (at least 1).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: vec![
                Slot {
                    used: SlotUse::Free,
                    id: SourceId(0),
                    started: 0,
                };
                capacity.max(1)
            ],
            next_id: 1,
            clock: 0,
        }
    }

    /// Allocate a slot for a new source. `finished` reports whether the
    /// backend says slot `i`'s current sound has stopped on its own
    /// (finished one-shots are reclaimed eagerly). Returns
    /// `(slot index, fresh id)`; `None` only when every slot holds a
    /// loop or a younger one-shot than everything stealable.
    #[must_use]
    pub fn allocate(
        &mut self,
        looping: bool,
        mut finished: impl FnMut(usize) -> bool,
    ) -> Option<(usize, SourceId)> {
        self.clock += 1;
        // Reclaim finished one-shots first.
        for i in 0..self.slots.len() {
            if self.slots[i].used == SlotUse::OneShot && finished(i) {
                self.slots[i].used = SlotUse::Free;
            }
        }
        // A free slot, else steal the OLDEST one-shot.
        let idx = match self.slots.iter().position(|s| s.used == SlotUse::Free) {
            Some(i) => i,
            None => self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| s.used == SlotUse::OneShot)
                .min_by_key(|(_, s)| s.started)
                .map(|(i, _)| i)?,
        };
        let id = SourceId(self.next_id);
        self.next_id += 1;
        self.slots[idx] = Slot {
            used: if looping {
                SlotUse::Looping
            } else {
                SlotUse::OneShot
            },
            id,
            started: self.clock,
        };
        Some((idx, id))
    }

    /// The slot currently owned by `id`, if it hasn't been stolen or
    /// stopped since.
    #[must_use]
    pub fn slot_of(&self, id: SourceId) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| s.used != SlotUse::Free && s.id == id)
    }

    /// Free `id`'s slot (explicit stop). Stale ids are ignored.
    pub fn release(&mut self, id: SourceId) -> Option<usize> {
        let i = self.slot_of(id)?;
        self.slots[i].used = SlotUse::Free;
        Some(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_reuses_freed_and_finished_slots() {
        let mut p = SourcePool::new(2);
        let (i0, a) = p.allocate(false, |_| false).expect("slot 0");
        let (i1, _b) = p.allocate(false, |_| false).expect("slot 1");
        assert_ne!(i0, i1);
        // Full, nothing finished ⇒ steals the OLDEST one-shot (a).
        let (i2, c) = p.allocate(false, |_| false).expect("steal oldest");
        assert_eq!(i2, i0);
        assert_eq!(p.slot_of(a), None, "stolen id is stale");
        assert_eq!(p.slot_of(c), Some(i0));
        // Finished slots are reclaimed before stealing.
        let (i3, _d) = p.allocate(false, |i| i == i1).expect("reclaim finished");
        assert_eq!(i3, i1);
    }

    #[test]
    fn loops_are_never_stolen() {
        let mut p = SourcePool::new(2);
        let (_, l0) = p.allocate(true, |_| false).expect("loop 0");
        let (_, _l1) = p.allocate(true, |_| false).expect("loop 1");
        assert!(
            p.allocate(false, |_| false).is_none(),
            "a pool full of loops refuses one-shots"
        );
        // Explicit stop frees the slot for the next allocation.
        let freed = p.release(l0).expect("live loop releases");
        let (i, _) = p.allocate(false, |_| false).expect("freed slot");
        assert_eq!(i, freed);
        assert!(p.release(l0).is_none(), "double release is a no-op");
    }

    /// A device-free [`AudioOut`] mock over a real [`SourcePool`], to
    /// pin the trait contract the (untestable, device-bound) kira
    /// backend follows: register keys, play/stop through the pool, and
    /// ignore calls on stale ids.
    #[derive(Default)]
    struct MockOut {
        sounds: usize,
        pool: Option<SourcePool>,
        applied: Vec<SourceId>,
        stopped: Vec<SourceId>,
    }

    impl AudioOut for MockOut {
        fn register(&mut self, _sound: &SoundBuffer) -> SoundKey {
            let k = SoundKey(self.sounds);
            self.sounds += 1;
            k
        }
        fn set_listener(&mut self, _pos: glam::DVec3, _orientation: glam::DQuat) {}
        fn play(
            &mut self,
            _sound: SoundKey,
            _at: glam::DVec3,
            _initial: Option<&SourceAcoustics>,
        ) -> Option<SourceId> {
            let p = self.pool.get_or_insert_with(|| SourcePool::new(2));
            p.allocate(false, |_| false).map(|(_, id)| id)
        }
        fn play_loop(
            &mut self,
            _sound: SoundKey,
            _at: glam::DVec3,
            _initial: Option<&SourceAcoustics>,
        ) -> Option<SourceId> {
            let p = self.pool.get_or_insert_with(|| SourcePool::new(2));
            p.allocate(true, |_| false).map(|(_, id)| id)
        }
        fn stop(&mut self, id: SourceId) {
            if let Some(p) = self.pool.as_mut() {
                if p.release(id).is_some() {
                    self.stopped.push(id);
                }
            }
        }
        fn set_source_position(&mut self, _id: SourceId, _pos: glam::DVec3) {}
        fn apply_source(&mut self, id: SourceId, _a: &SourceAcoustics) {
            // Contract: only act on a live id.
            if self.pool.as_ref().and_then(|p| p.slot_of(id)).is_some() {
                self.applied.push(id);
            }
        }
        fn apply_listener(&mut self, _a: &ListenerAcoustics) {}
    }

    #[test]
    fn audio_out_contract_via_mock() {
        use crate::{AcousticsConfig, SoundBuffer, SourceAcoustics};
        // AudioOut is object-safe (hosts store `Box<dyn AudioOut>`).
        let _obj: Box<dyn AudioOut> = Box::new(MockOut::default());

        let buf = SoundBuffer {
            sample_rate: 8,
            samples: vec![0.0; 8],
        };
        let mut out = MockOut::default();
        let k0 = out.register(&buf);
        let k1 = out.register(&buf);
        assert_ne!(k0, k1, "distinct sound keys");

        let a = out.play(k0, glam::DVec3::ZERO, None).expect("first voice");
        let b = out.play(k1, glam::DVec3::ZERO, None).expect("second voice");
        // Applying to live sources is honoured.
        let clear = SourceAcoustics::clear(&AcousticsConfig::default());
        out.apply_source(a, &clear);
        out.apply_source(b, &clear);
        // Stop `a`, then a stale apply on `a` must be ignored.
        out.stop(a);
        out.apply_source(a, &clear);

        assert_eq!(out.stopped, vec![a]);
        assert_eq!(out.applied, vec![a, b], "stale apply on `a` was dropped");
    }
}

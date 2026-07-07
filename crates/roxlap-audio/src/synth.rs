//! AU.2 — procedurally synthesized demo sounds (house pattern: no
//! binary assets; the Doom demo synthesizes its GIFs, we synthesize
//! our bleeps). Deterministic — the noise source is a fixed LCG — so
//! buffers are testable byte-for-byte and identical across runs.
//!
//! Retro engine, retro bleeps: these read as programmer art on
//! purpose. Real assets can replace them at the playback layer (the
//! kira backend also accepts encoded files via its own loaders).

/// A mono PCM buffer in `-1..=1` floats — the backend-agnostic sound
/// asset the [`crate::AudioOut`] trait registers.
#[derive(Debug, Clone, PartialEq)]
pub struct SoundBuffer {
    /// Samples per second.
    pub sample_rate: u32,
    /// Mono samples, `-1..=1`.
    pub samples: Vec<f32>,
}

impl SoundBuffer {
    /// Buffer duration in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let n = self.samples.len() as f32;
        #[allow(clippy::cast_precision_loss)]
        let sr = self.sample_rate as f32;
        n / sr
    }
}

/// Fixed-seed LCG noise in `-1..=1` (Numerical Recipes constants) —
/// deterministic across runs and platforms.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Top 24 bits → [-1, 1).
        #[allow(clippy::cast_precision_loss)]
        let v = ((self.0 >> 40) as f32) / 8_388_608.0;
        v - 1.0
    }
}

/// A plasma-shot transient: a sharp noise burst with a fast downward
/// pitch sweep, ~120 ms.
#[must_use]
pub fn shot(sample_rate: u32) -> SoundBuffer {
    let n = (sample_rate as usize * 120) / 1000;
    let mut rng = Lcg(0x5EED_0001);
    let mut samples = Vec::with_capacity(n);
    #[allow(clippy::cast_precision_loss)]
    let sr = sample_rate as f32;
    let mut phase = 0.0f32;
    for i in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f32 / sr;
        // 1400 Hz → 300 Hz sweep under an exponential decay, plus a
        // noise tail for the "crack".
        let f = 1400.0 * (-t * 18.0).exp() + 300.0;
        phase += std::f32::consts::TAU * f / sr;
        let env = (-t * 30.0).exp();
        let s = (phase.sin() * 0.7 + rng.next() * 0.3) * env;
        samples.push(s.clamp(-1.0, 1.0));
    }
    SoundBuffer {
        sample_rate,
        samples,
    }
}

/// An impact boom: a low sine thump plus a noise tail, ~500 ms —
/// the carve explosion.
#[must_use]
pub fn impact(sample_rate: u32) -> SoundBuffer {
    let n = (sample_rate as usize * 500) / 1000;
    let mut rng = Lcg(0x5EED_0002);
    let mut samples = Vec::with_capacity(n);
    #[allow(clippy::cast_precision_loss)]
    let sr = sample_rate as f32;
    let mut phase = 0.0f32;
    for i in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f32 / sr;
        let f = 120.0 * (-t * 6.0).exp() + 45.0;
        phase += std::f32::consts::TAU * f / sr;
        let thump = phase.sin() * (-t * 7.0).exp();
        let rumble = rng.next() * (-t * 4.0).exp() * 0.35;
        samples.push((thump * 0.8 + rumble).clamp(-1.0, 1.0));
    }
    SoundBuffer {
        sample_rate,
        samples,
    }
}

/// A crystal hum: a softly beating detuned sine pair, seamlessly
/// loopable (both partials complete an integer number of cycles).
#[must_use]
pub fn hum(sample_rate: u32) -> SoundBuffer {
    // Loop length 1 s ⇒ integer frequencies loop seamlessly. 220 Hz +
    // 223 Hz beats at 3 Hz — a slow shimmer.
    let n = sample_rate as usize;
    let mut samples = Vec::with_capacity(n);
    #[allow(clippy::cast_precision_loss)]
    let sr = sample_rate as f32;
    for i in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f32 / sr;
        let a = (std::f32::consts::TAU * 220.0 * t).sin();
        let b = (std::f32::consts::TAU * 223.0 * t).sin();
        samples.push((a + b) * 0.25);
    }
    SoundBuffer {
        sample_rate,
        samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    #[test]
    fn buffers_have_expected_shapes() {
        let s = shot(SR);
        assert!((s.duration() - 0.12).abs() < 0.01);
        let i = impact(SR);
        assert!((i.duration() - 0.5).abs() < 0.01);
        let h = hum(SR);
        assert!((h.duration() - 1.0).abs() < 0.01);
        for b in [&s, &i, &h] {
            assert!(!b.samples.is_empty());
            assert!(
                b.samples.iter().all(|v| (-1.0..=1.0).contains(v)),
                "samples stay in range"
            );
            assert!(
                b.samples.iter().any(|v| v.abs() > 0.05),
                "buffer is not silence"
            );
        }
    }

    #[test]
    fn hum_loops_seamlessly() {
        // Both partials are integer Hz over a 1 s buffer, so the loop
        // wrap (last → first sample) continues the waveform: the jump
        // must be no bigger than a typical adjacent-sample step.
        let h = hum(SR);
        let wrap = (h.samples[0] - h.samples[h.samples.len() - 1]).abs();
        let step = (h.samples[1] - h.samples[0]).abs();
        assert!(
            wrap <= step * 2.0 + 1e-4,
            "loop wrap jump {wrap} vs adjacent step {step}"
        );
    }

    #[test]
    fn deterministic() {
        assert_eq!(shot(SR), shot(SR));
        assert_eq!(impact(SR), impact(SR));
        assert_eq!(hum(SR), hum(SR));
    }
}

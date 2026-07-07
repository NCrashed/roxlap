//! AU.2 — the [`kira`] 0.12 playback backend for [`AudioOut`].
//!
//! Topology (entry doc decision 2): one shared [`SendTrack`] carrying a
//! [`Reverb`] whose feedback/damping/mix tween at runtime, and a fixed
//! **pool** of [`SpatialTrack`]s (hazard 2: allocate up front, reuse) —
//! each carrying its own lowpass [`Filter`] and a per-source send into
//! the reverb. The acoustics core hands us parameters; we apply them
//! with tweens (hazard 3: never raw jumps).
//!
//! Every `kira` type stays inside this module — the rest of the crate,
//! and every host, compiles without the feature (hazard 1: kira's API
//! churns).

use std::sync::Arc;
use std::time::Duration;

use glam::{DQuat, DVec3};
use kira::{
    effect::{
        filter::{FilterBuilder, FilterHandle, FilterMode},
        reverb::{ReverbBuilder, ReverbHandle},
    },
    sound::static_sound::{StaticSoundData, StaticSoundHandle, StaticSoundSettings},
    track::{
        SendTrackBuilder, SendTrackHandle, SpatialTrackBuilder, SpatialTrackDistances,
        SpatialTrackHandle,
    },
    AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Easing, Frame, Mix, Tween,
};

use crate::backend::{AudioOut, SourceId, SourcePool};
use crate::synth::SoundBuffer;
use crate::{ListenerAcoustics, SoundKey, SourceAcoustics};

/// Fast tween for per-frame parameters — per-source occlusion (gain,
/// cutoff, send) and the listener pose — short so a moving listener
/// tracks smoothly without zippering.
const FAST_TWEEN_MS: u64 = 120;
/// Slow tween for the reverb *environment* (feedback, damping, send
/// level) — the room's character eases in over ~a second (the estimator
/// already smooths, this is the final anti-zipper).
const ENV_TWEEN_MS: u64 = 1000;
/// Very short fade when a playing voice is stolen, so the cut doesn't
/// click.
const STEAL_TWEEN_MS: u64 = 15;
/// Default spatial-track pool size. Sources beyond this steal the
/// oldest finished/one-shot slot.
const DEFAULT_POOL: usize = 24;

fn tween(ms: u64) -> Tween {
    Tween {
        duration: Duration::from_millis(ms),
        easing: Easing::Linear,
        ..Default::default()
    }
}

/// Grid-local world position → kira's `mint` f32 vector. **Caveat:**
/// this truncates f64 world coords to f32 — at coordinates in the
/// thousands the spatialization position is accurate to ~centimetres,
/// which is inaudible; very large worlds (10^6+ units) would want a
/// listener-relative rebase first.
fn mint_pos(p: DVec3) -> mint::Vector3<f32> {
    p.as_vec3().into()
}

/// Linear reverb-return amplitude (`0..=1`) → send-track volume. On an
/// aux send the reverb runs fully wet, so the room's wetness is the
/// send LEVEL, not a dry/wet blend: `0` ⇒ silent return (dry room),
/// louder ⇒ wetter.
fn mix_to_db(mix: f32) -> Decibels {
    if mix <= 1e-4 {
        Decibels::SILENCE
    } else {
        Decibels((20.0 * mix.log10()).max(Decibels::SILENCE.0))
    }
}

/// One pooled spatial track: its handle, its lowpass filter handle,
/// and the currently-playing sound (kept so loops can be stopped — the
/// spatial track itself has no `stop`).
struct PoolTrack {
    track: SpatialTrackHandle,
    filter: FilterHandle,
    sound: Option<StaticSoundHandle>,
}

/// A [`kira`] 0.12 playback backend. Construct with [`KiraAudio::new`]
/// (opens the default output device) and drive it through [`AudioOut`].
pub struct KiraAudio {
    // Ownership anchor: dropping the manager tears down the audio
    // thread. Never called after construction — kept alive only.
    #[allow(dead_code)]
    manager: AudioManager<DefaultBackend>,
    /// The reverb aux send — its VOLUME is the room wetness (the reverb
    /// itself runs fully wet on the bus). Driven by `apply_listener`.
    send: SendTrackHandle,
    listener: kira::listener::ListenerHandle,
    send_id: kira::track::SendTrackId,
    reverb: ReverbHandle,
    tracks: Vec<PoolTrack>,
    pool: SourcePool,
    sounds: Vec<StaticSoundData>,
}

impl KiraAudio {
    /// Open the default audio device and build the reverb send + the
    /// spatial-track pool (24 voices; [`with_capacity`](Self::with_capacity)
    /// for another size).
    ///
    /// # Errors
    /// Propagates the backend error if the device or any mixer resource
    /// can't be created.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_capacity(DEFAULT_POOL)
    }

    /// [`KiraAudio::new`] with an explicit voice-pool size.
    ///
    /// # Errors
    /// As [`KiraAudio::new`].
    pub fn with_capacity(pool: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())?;
        let listener = manager.add_listener(
            mint_pos(DVec3::ZERO),
            mint::Quaternion::from([0.0, 0.0, 0.0, 1.0]),
        )?;

        // One shared reverb on an aux send. The reverb runs FULLY WET
        // (`Mix(1.0)`) because it's a parallel send bus — any dry
        // component here would re-add a copy of each source's dry
        // signal to the master (louder + comb-filtered, the classic
        // aux-bus mistake). Room wetness is the send-track VOLUME
        // instead, which starts silent (dry) and rises in enclosed
        // spaces via `apply_listener`.
        let mut send_builder = SendTrackBuilder::new().volume(Decibels::SILENCE);
        let reverb = send_builder.add_effect(
            ReverbBuilder::new()
                .feedback(0.6)
                .damping(0.4)
                .mix(Mix(1.0)),
        );
        let send = manager.add_send_track(send_builder)?;
        let send_id = send.id();

        // Spatial attenuation range: intentionally INSIDE the occlusion
        // budget (`AcousticsConfig::max_distance`, default 128) — a
        // source is fully distance-attenuated by 96 before occlusion
        // stops marching at 128.
        let distances = SpatialTrackDistances {
            min_distance: 1.0,
            max_distance: 96.0,
        };

        let mut tracks = Vec::with_capacity(pool.max(1));
        for _ in 0..pool.max(1) {
            let mut builder = SpatialTrackBuilder::new()
                .distances(distances)
                .with_send(send_id, Decibels(0.0));
            let filter = builder.add_effect(
                FilterBuilder::new()
                    .mode(FilterMode::LowPass)
                    .cutoff(20_000.0),
            );
            let track = manager.add_spatial_sub_track(&listener, mint_pos(DVec3::ZERO), builder)?;
            tracks.push(PoolTrack {
                track,
                filter,
                sound: None,
            });
        }

        let pool_len = tracks.len();
        Ok(Self {
            manager,
            send,
            listener,
            send_id,
            reverb,
            tracks,
            pool: SourcePool::new(pool_len),
            sounds: Vec::new(),
        })
    }

    /// Convert a mono [`SoundBuffer`] to kira's stereo-frame static
    /// data (mono duped to both channels). Looping is applied per play
    /// in [`start`](Self::start), not baked here.
    fn to_static(sound: &SoundBuffer) -> StaticSoundData {
        let frames: Arc<[Frame]> = sound.samples.iter().map(|&s| Frame::new(s, s)).collect();
        StaticSoundData {
            sample_rate: sound.sample_rate,
            frames,
            settings: StaticSoundSettings::default(),
            slice: None,
        }
    }

    /// Shared body of `play` / `play_loop`: allocate a pool slot, point
    /// its track at `at`, reset its filter, and start the sound.
    fn start(&mut self, sound: SoundKey, at: DVec3, looping: bool) -> Option<SourceId> {
        let data = self.sounds.get(sound.0)?.clone();
        let tracks = &self.tracks;
        let (idx, id) = self.pool.allocate(looping, |i| {
            // A one-shot slot is finished when its track has no sounds.
            tracks[i].track.num_sounds() == 0
        })?;
        let data = if looping { data.loop_region(..) } else { data };
        let slot = &mut self.tracks[idx];
        // If this slot was STOLEN from a still-playing voice, silence
        // the old sound first — dropping a `StaticSoundHandle` does NOT
        // stop it in kira, so "steal" must actually cut (a quick fade
        // to avoid a click). A finished/reused voice's handle is a
        // harmless no-op.
        if let Some(mut old) = slot.sound.take() {
            old.stop(tween(STEAL_TWEEN_MS));
        }
        slot.track.set_position(mint_pos(at), tween(0));
        // Reset occlusion state to "clear" for the new source.
        slot.track.set_volume(Decibels(0.0), tween(0));
        slot.filter.set_cutoff(20_000.0, tween(0));
        let _ = slot.track.set_send(self.send_id, Decibels(0.0), tween(0));
        slot.sound = slot.track.play(data).ok();
        slot.sound.as_ref()?;
        Some(id)
    }
}

impl AudioOut for KiraAudio {
    fn register(&mut self, sound: &SoundBuffer) -> SoundKey {
        self.sounds.push(Self::to_static(sound));
        SoundKey(self.sounds.len() - 1)
    }

    fn set_listener(&mut self, pos: DVec3, orientation: DQuat) {
        let q = orientation.as_quat();
        self.listener
            .set_position(mint_pos(pos), tween(FAST_TWEEN_MS));
        self.listener.set_orientation(
            mint::Quaternion::from([q.x, q.y, q.z, q.w]),
            tween(FAST_TWEEN_MS),
        );
    }

    fn play(&mut self, sound: SoundKey, at: DVec3) -> Option<SourceId> {
        self.start(sound, at, false)
    }

    fn play_loop(&mut self, sound: SoundKey, at: DVec3) -> Option<SourceId> {
        self.start(sound, at, true)
    }

    fn stop(&mut self, id: SourceId) {
        if let Some(idx) = self.pool.release(id) {
            if let Some(sound) = self.tracks[idx].sound.as_mut() {
                sound.stop(tween(FAST_TWEEN_MS));
            }
            self.tracks[idx].sound = None;
        }
    }

    fn set_source_position(&mut self, id: SourceId, pos: DVec3) {
        if let Some(idx) = self.pool.slot_of(id) {
            self.tracks[idx]
                .track
                .set_position(mint_pos(pos), tween(FAST_TWEEN_MS));
        }
    }

    fn apply_source(&mut self, id: SourceId, acoustics: &SourceAcoustics) {
        let Some(idx) = self.pool.slot_of(id) else {
            return;
        };
        let t = tween(FAST_TWEEN_MS);
        let slot = &mut self.tracks[idx];
        slot.track.set_volume(Decibels(acoustics.gain_db), t);
        slot.filter
            .set_cutoff(f64::from(acoustics.lowpass_cutoff_hz), t);
        let _ = slot
            .track
            .set_send(self.send_id, Decibels(acoustics.reverb_send_db), t);
    }

    fn apply_listener(&mut self, acoustics: &ListenerAcoustics) {
        let t = tween(ENV_TWEEN_MS);
        // Reverb CHARACTER (decay/damping) tweens on the effect; room
        // WETNESS is the aux-send level (the reverb itself stays fully
        // wet — see `KiraAudio::with_capacity`).
        self.reverb
            .set_feedback(f64::from(acoustics.reverb_feedback), t);
        self.reverb
            .set_damping(f64::from(acoustics.reverb_damping), t);
        self.send.set_volume(mix_to_db(acoustics.reverb_mix), t);
    }
}

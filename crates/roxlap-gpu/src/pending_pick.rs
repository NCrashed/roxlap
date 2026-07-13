//! PW.1 — the platform-neutral half of async (one-frame-latency)
//! depth picking for the wasm GPU path.
//!
//! WebGPU has no blocking buffer readback: `map_async` resolves on
//! browser event-loop turns, never inside the call that issued it. So
//! on wasm `pick_depth` becomes a small state machine: a call SUBMITS
//! the readback for its own pixel and returns the LATEST COMPLETED
//! result — usually `None` on the first click and the value on the
//! next call (one-frame latency). [`PendingPick`] is that state
//! machine, kept free of wgpu types so it unit-tests without a
//! device; the driver ([`crate::GpuRenderer::read_depth_pixel_async`])
//! owns the buffers and the `map_async` callback.

use std::sync::{Arc, Mutex};

/// The async-pick state machine. One readback in flight at a time:
/// requests arriving while one is mapping are coalesced away (the
/// next call re-arms with its own, newest coordinates, so a burst of
/// clicks costs one readback and the last click wins).
#[derive(Debug, Default)]
pub(crate) struct PendingPick {
    /// Pixel whose readback is currently mapping, if any.
    in_flight: Option<(u32, u32)>,
    /// The most recently completed pick: its pixel + depth (`None`
    /// depth = sky / no-hit / readback error).
    completed: Option<((u32, u32), Option<f32>)>,
}

impl PendingPick {
    /// A pick request for pixel `(x, y)`: `true` when the driver must
    /// submit a readback for it now (nothing was in flight); `false`
    /// while a readback is mapping — this click is coalesced away.
    pub(crate) fn request(&mut self, x: u32, y: u32) -> bool {
        if self.in_flight.is_some() {
            return false;
        }
        self.in_flight = Some((x, y));
        true
    }

    /// The driver observed the in-flight map completing: record its
    /// result (`None` = sky / no-hit / map error). No-op when nothing
    /// is in flight (defensive; the driver only calls this in flight).
    pub(crate) fn complete(&mut self, depth: Option<f32>) {
        if let Some(px) = self.in_flight.take() {
            self.completed = Some((px, depth));
        }
    }

    /// The latest completed pick — its pixel + depth. The pixel may be
    /// an EARLIER click's than the one just requested: that is the
    /// one-frame-latency contract.
    pub(crate) fn latest(&self) -> Option<((u32, u32), Option<f32>)> {
        self.completed
    }

    /// True while a readback is mapping.
    pub(crate) fn is_in_flight(&self) -> bool {
        self.in_flight.is_some()
    }
}

/// The driver-side state the machine is embedded in: the shared cell
/// the `map_async` callback writes, and the per-pick 4-byte staging
/// buffer (owned here, NOT the shared `depth_readback`, so a resize /
/// scene swap between calls cannot invalidate an in-flight pick — the
/// copy already executed against the depth buffer at submit time).
#[derive(Debug)]
pub(crate) struct AsyncPickState {
    pub(crate) pending: PendingPick,
    /// Filled by the in-flight readback's `map_async` callback. A
    /// fresh `Arc` per submission — a hypothetical late callback from
    /// an abandoned pick writes into its own orphaned cell.
    pub(crate) map_result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>,
    /// The in-flight pick's staging buffer; dropped (auto-unmapped)
    /// at harvest.
    pub(crate) staging: Option<wgpu::Buffer>,
}

impl Default for AsyncPickState {
    fn default() -> Self {
        Self {
            pending: PendingPick::default(),
            map_result: Arc::new(Mutex::new(None)),
            staging: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PendingPick;

    // The canonical click sequence: request → in flight → complete →
    // the NEXT request returns the completed value and re-arms.
    #[test]
    fn request_complete_rearm_cycle() {
        let mut p = PendingPick::default();
        assert!(p.latest().is_none(), "nothing completed yet");
        assert!(p.request(3, 4), "idle: submit");
        assert!(p.is_in_flight());
        assert!(p.latest().is_none(), "still nothing completed");

        p.complete(Some(12.5));
        assert!(!p.is_in_flight());
        assert_eq!(p.latest(), Some(((3, 4), Some(12.5))));

        // Re-arm: the new request submits; the old result stays
        // readable until the new one completes.
        assert!(p.request(7, 8), "idle again: submit");
        assert_eq!(p.latest(), Some(((3, 4), Some(12.5))));
        p.complete(None); // the new pixel was sky
        assert_eq!(p.latest(), Some(((7, 8), None)));
    }

    // Clicks while a readback is mapping are coalesced away — exactly
    // one readback in flight, no queue to drain.
    #[test]
    fn in_flight_coalesces_further_requests() {
        let mut p = PendingPick::default();
        assert!(p.request(1, 1));
        assert!(!p.request(2, 2), "in flight: coalesced");
        assert!(!p.request(3, 3), "still coalesced");
        p.complete(Some(5.0));
        // The completed pick is the ORIGINAL one; the coalesced
        // clicks never happened.
        assert_eq!(p.latest(), Some(((1, 1), Some(5.0))));
        assert!(p.request(4, 4), "idle again after harvest");
    }

    // A completion with nothing in flight must not fabricate a result.
    #[test]
    fn complete_without_in_flight_is_noop() {
        let mut p = PendingPick::default();
        p.complete(Some(9.0));
        assert!(p.latest().is_none());
    }

    // A sky/no-hit completion is a real completion (None depth), not
    // a stuck in-flight state.
    #[test]
    fn sky_completion_clears_in_flight() {
        let mut p = PendingPick::default();
        assert!(p.request(0, 0));
        p.complete(None);
        assert!(!p.is_in_flight());
        assert_eq!(p.latest(), Some(((0, 0), None)));
    }
}

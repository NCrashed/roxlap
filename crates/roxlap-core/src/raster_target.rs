//! `RasterTarget` — a `Copy` borrowed view of the framebuffer +
//! zbuffer as raw pointers, the compositing primitive shared by the
//! DDA terrain renderer ([`crate::dda`]) and the DDA sprite raycaster
//! ([`crate::dda_sprite`]).
//!
//! Holding `&'a mut [u32]` / `&'a mut [f32]` directly would force an
//! exclusive borrow per instance, blocking the renderer's tile bands
//! from running on multiple threads even though their pixel writes are
//! disjoint. `RasterTarget` is constructed safely from exclusive slice
//! borrows, then re-exposes the memory as raw pointers tied to lifetime
//! `'a` via `PhantomData`. Once it exists it is the sole path to the
//! underlying memory for the duration of `'a`.
//!
//! # Safety contract for parallel use
//! Callers that copy a `RasterTarget` and pass copies to multiple
//! threads MUST guarantee the threads collectively write to
//! pairwise-disjoint pixel indices. The parallel DDA driver enforces
//! this via per-band row ranges; single-threaded callers hold one copy
//! and trivially satisfy the invariant.

use std::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
pub struct RasterTarget<'a> {
    fb_ptr: *mut u32,
    fb_len: usize,
    zb_ptr: *mut f32,
    zb_len: usize,
    _marker: PhantomData<&'a mut [u32]>,
}

// SAFETY: `RasterTarget` is morally a borrowed mutable slice pair —
// the same shape `&'a mut [u32]` / `&'a mut [f32]` would have, both of
// which are `Send` when `T: Send`. Multi-thread safety is enforced
// by the wedge / strip-disjoint write invariant (see struct doc).
unsafe impl Send for RasterTarget<'_> {}

// SAFETY: sharing `&RasterTarget` across threads exposes only the
// raw pointers + lengths. Reading a pointer field is itself free of
// data races; concurrent writes through the pointer are gated by
// the disjoint-write invariant the caller upholds. Required so
// `ScalarRasterizer: Sync`, which `rayon::par_iter_mut` needs to
// share `&rasterizer` across the strip-parallel closures (R12.3.1).
unsafe impl Sync for RasterTarget<'_> {}

impl<'a> RasterTarget<'a> {
    /// Build a target from exclusive slice borrows. The slices are
    /// consumed (their `&'a mut` reborrow is the load-bearing thing —
    /// this constructor is the only way to mint a `RasterTarget`
    /// from safe code).
    #[must_use]
    pub fn new(framebuffer: &'a mut [u32], zbuffer: &'a mut [f32]) -> Self {
        Self {
            fb_ptr: framebuffer.as_mut_ptr(),
            fb_len: framebuffer.len(),
            zb_ptr: zbuffer.as_mut_ptr(),
            zb_len: zbuffer.len(),
            _marker: PhantomData,
        }
    }

    /// Framebuffer length in `u32` elements.
    #[must_use]
    pub fn fb_len(self) -> usize {
        self.fb_len
    }

    /// Raw mutable framebuffer pointer. Used by SSE blocks that do
    /// their own arithmetic + bounds reasoning.
    ///
    /// # Safety
    /// Callers must respect `fb_len` and the parallel-use invariant.
    #[must_use]
    pub fn fb_ptr(self) -> *mut u32 {
        self.fb_ptr
    }

    /// Raw mutable zbuffer pointer. Same contract as
    /// [`Self::fb_ptr`].
    #[must_use]
    pub fn zb_ptr(self) -> *mut f32 {
        self.zb_ptr
    }

    /// Write one ARGB pixel.
    ///
    /// # Safety
    /// `idx < self.fb_len()`, plus the parallel-use invariant.
    pub unsafe fn write_color(self, idx: usize, color: u32) {
        debug_assert!(idx < self.fb_len, "fb idx {} >= len {}", idx, self.fb_len);
        // SAFETY: caller asserts in-bounds + disjoint-from-other-threads.
        unsafe { self.fb_ptr.add(idx).write(color) };
    }

    /// Write one z-buffer entry.
    ///
    /// # Safety
    /// `idx < self.fb_len()` (zbuffer length matches fb), plus the
    /// parallel-use invariant.
    pub unsafe fn write_depth(self, idx: usize, z: f32) {
        debug_assert!(idx < self.zb_len, "zb idx {} >= len {}", idx, self.zb_len);
        // SAFETY: caller asserts in-bounds + disjoint-from-other-threads.
        unsafe { self.zb_ptr.add(idx).write(z) };
    }

    /// Read one z-buffer entry — the terrain depth a translucent sprite
    /// march tests against (TV stage).
    ///
    /// # Safety
    /// `idx < self.fb_len()`, plus the parallel-use invariant.
    #[must_use]
    pub unsafe fn read_depth(self, idx: usize) -> f32 {
        debug_assert!(idx < self.zb_len, "zb idx {} >= len {}", idx, self.zb_len);
        // SAFETY: caller asserts in-bounds + disjoint-from-other-threads.
        unsafe { *self.zb_ptr.add(idx) }
    }

    /// Read one framebuffer pixel — the background a translucent sprite
    /// composites over when its ray exits without an opaque hit (TV stage).
    ///
    /// # Safety
    /// `idx < self.fb_len()`, plus the parallel-use invariant.
    #[must_use]
    pub unsafe fn read_color(self, idx: usize) -> u32 {
        debug_assert!(idx < self.fb_len, "fb idx {} >= len {}", idx, self.fb_len);
        // SAFETY: caller asserts in-bounds + disjoint-from-other-threads.
        unsafe { *self.fb_ptr.add(idx) }
    }

    /// Depth-tested write: store `(color, z)` only if `z` is strictly
    /// closer (smaller) than the current z-buffer entry. Returns whether
    /// the pixel was written. The compositing primitive the DDA sprite
    /// raycaster uses to occlude sprites against terrain.
    ///
    /// # Safety
    /// `idx < self.fb_len()`, plus the parallel-use invariant.
    #[must_use]
    pub unsafe fn z_test_write(self, idx: usize, color: u32, z: f32) -> bool {
        debug_assert!(idx < self.zb_len, "zb idx {} >= len {}", idx, self.zb_len);
        // SAFETY: caller asserts in-bounds + disjoint-from-other-threads.
        unsafe {
            if z < *self.zb_ptr.add(idx) {
                self.fb_ptr.add(idx).write(color);
                self.zb_ptr.add(idx).write(z);
                true
            } else {
                false
            }
        }
    }
}

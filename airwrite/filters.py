"""Signal smoothing for the fingertip pointer.

Raw MediaPipe landmarks jitter by several pixels frame to frame. Writing with
an unfiltered fingertip produces illegible, furry strokes. A plain EMA fixes
the jitter but adds lag that makes fast strokes lag behind the finger and
corners round off.

The One Euro filter adapts its cutoff to speed: heavy smoothing when the
finger is nearly still (where jitter is visible), light smoothing when it
moves fast (where lag is visible). That trade is exactly what handwriting
needs.

Reference: Casiez, Roussel & Vogel, "1e Filter" (CHI 2012).
"""

import math


class _LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    """Adaptive low-pass filter for a single scalar channel.

    min_cutoff: lower -> smoother when still (more jitter removed, more lag)
    beta:       higher -> more responsive when moving fast (less lag)
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._prev_x = None
        self._prev_t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._prev_t is None:
            self._prev_t, self._prev_x = t, x
            self._x.y = x
            return x

        dt = t - self._prev_t
        # Guard against duplicate timestamps and clock hiccups; 60fps is a sane
        # fallback since dt only sets the filter's time constant.
        if dt <= 0 or dt > 1.0:
            dt = 1.0 / 60.0
        self._prev_t = t

        dx = (x - self._prev_x) / dt
        self._prev_x = x
        edx = self._dx(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, self._alpha(cutoff, dt))


class PointFilter:
    """One Euro filter over a 2D point."""

    def __init__(self, min_cutoff=1.0, beta=0.007):
        self._fx = OneEuroFilter(min_cutoff, beta)
        self._fy = OneEuroFilter(min_cutoff, beta)

    def __call__(self, x, y, t):
        return self._fx(x, t), self._fy(y, t)

"""Headless tests for the parts that do not need a camera or MediaPipe."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from airwrite.canvas import Canvas
from airwrite.filters import OneEuroFilter, PointFilter
from airwrite.gestures import (CLEAR, DRAW, ERASE, IDLE, GestureDebouncer,
                               classify, fingers_up)


def hand(thumb_out=False, index=False, middle=False, ring=False, pinky=False):
    """Build a fake landmark list with the requested fingers extended.

    Only the tip/PIP pairs the classifier reads are meaningful; y decreases
    upward in image coordinates, so an extended finger gets a smaller tip y.
    """
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    # Thumb: tip vs the joint below it, compared on x for a right hand.
    lm[3] = SimpleNamespace(x=0.50, y=0.5)
    lm[4] = SimpleNamespace(x=0.45 if thumb_out else 0.55, y=0.5)
    for tip, pip, extended in ((8, 6, index), (12, 10, middle),
                               (16, 14, ring), (20, 18, pinky)):
        lm[pip] = SimpleNamespace(x=0.5, y=0.50)
        lm[tip] = SimpleNamespace(x=0.5, y=0.40 if extended else 0.60)
    return lm


class TestGestures:
    def test_index_only_is_draw(self):
        assert classify(hand(index=True)) == DRAW

    def test_two_fingers_is_erase(self):
        assert classify(hand(index=True, middle=True)) == ERASE

    def test_closed_fist_is_clear(self):
        assert classify(hand()) == CLEAR

    def test_open_palm_is_idle(self):
        palm = hand(thumb_out=True, index=True, middle=True, ring=True,
                    pinky=True)
        assert classify(palm) == IDLE

    def test_thumb_direction_flips_with_handedness(self):
        lm = hand(thumb_out=True)
        assert fingers_up(lm, "Right")[0] is True
        assert fingers_up(lm, "Left")[0] is False


class TestDebouncer:
    def test_holds_until_stable(self):
        d = GestureDebouncer(window=3)
        assert d.update(DRAW) != DRAW      # not yet enough frames
        assert d.update(DRAW) != DRAW
        assert d.update(DRAW) == DRAW

    def test_single_bad_frame_does_not_lift_the_pen(self):
        d = GestureDebouncer(window=3)
        for _ in range(3):
            d.update(DRAW)
        assert d.update(IDLE) == DRAW      # glitch frame absorbed
        assert d.update(DRAW) == DRAW


class TestOneEuro:
    def test_suppresses_jitter_around_a_still_point(self):
        f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
        rng = np.random.default_rng(0)
        noisy = [100.0 + rng.normal(0, 3) for _ in range(120)]
        out = [f(v, i / 30.0) for i, v in enumerate(noisy)]
        # Compare the settled tail, past the filter's warm-up.
        assert np.std(out[40:]) < np.std(noisy[40:]) / 3

    def test_tracks_a_fast_ramp_without_falling_far_behind(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        out = [f(float(i * 10), i / 30.0) for i in range(60)]
        assert abs(out[-1] - 590.0) < 25.0

    def test_survives_duplicate_timestamps(self):
        f = OneEuroFilter()
        f(10.0, 1.0)
        assert math.isfinite(f(20.0, 1.0))   # dt == 0 must not divide by zero

    def test_point_filter_smooths_both_axes(self):
        pf = PointFilter()
        pf(0.0, 0.0, 0.0)
        x, y = pf(100.0, 100.0, 1 / 30.0)
        assert 0.0 < x < 100.0 and 0.0 < y < 100.0


class TestCanvas:
    def test_pen_up_discards_single_point_strokes(self):
        c = Canvas(640, 480)
        c.pen_down((10, 10))
        c.pen_up()
        assert c.strokes == []

    def test_pen_cycle_creates_separate_strokes(self):
        c = Canvas(640, 480)
        for p in [(1, 1), (2, 2)]:
            c.pen_down(p)
        c.pen_up()
        for p in [(50, 50), (60, 60)]:
            c.pen_down(p)
        c.pen_up()
        assert len(c.strokes) == 2

    def test_duplicate_samples_are_dropped(self):
        c = Canvas(640, 480)
        for _ in range(5):
            c.pen_down((10, 10))
        c.pen_down((11, 11))
        assert len(c.strokes[0]) == 2

    def test_erase_removes_only_the_stroke_under_the_cursor(self):
        c = Canvas(640, 480)
        for p in [(10, 10), (20, 20)]:
            c.pen_down(p)
        c.pen_up()
        for p in [(400, 400), (410, 410)]:
            c.pen_down(p)
        c.pen_up()
        assert c.erase_near((15, 15), radius=30) == 1
        assert len(c.strokes) == 1
        assert c.strokes[0].points[0] == (400, 400)

    def test_undo_drops_the_last_stroke(self):
        c = Canvas(640, 480)
        for p in [(1, 1), (2, 2)]:
            c.pen_down(p)
        c.pen_up()
        c.undo()
        assert c.strokes == []

    def test_ocr_render_is_black_ink_on_white_at_target_height(self):
        c = Canvas(640, 480)
        for p in [(100, 100), (100, 200), (180, 200)]:   # an "L"
            c.pen_down(p)
        c.pen_up()

        img = c.render_for_ocr(target_height=128)
        assert img is not None
        assert img.shape[0] == 128
        assert img.dtype == np.uint8
        assert img.max() == 255          # white background survives
        assert img.min() < 60            # dark ink is present
        # Ink should be a minority of the crop; a mostly-black image means the
        # polarity got inverted somewhere.
        assert (img < 128).mean() < 0.5

    def test_ocr_render_crops_to_the_ink(self):
        c = Canvas(1280, 720)
        for p in [(600, 300), (620, 320)]:
            c.pen_down(p)
        c.pen_up()
        img = c.render_for_ocr(target_height=64, padding=10)
        # Width tracks the ~40px padded bounding box, not the 1280px frame.
        assert img.shape[1] < 200

    def test_empty_canvas_renders_nothing(self):
        assert Canvas(640, 480).render_for_ocr() is None

    def test_is_empty_ignores_stray_dots(self):
        c = Canvas(640, 480)
        c.pen_down((5, 5))
        assert c.is_empty()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

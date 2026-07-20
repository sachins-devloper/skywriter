"""Stroke storage and rendering."""

import cv2
import numpy as np

# Palette cycled by the color key. BGR, since OpenCV.
COLORS = [
    ("cyan", (255, 255, 0)),
    ("green", (0, 255, 0)),
    ("magenta", (255, 0, 255)),
    ("yellow", (0, 255, 255)),
]


class Stroke:
    def __init__(self, color, thickness):
        self.points = []
        self.color = color
        self.thickness = thickness

    def add(self, point):
        # Skip duplicate samples; they add no shape and bloat the export.
        if self.points and self.points[-1] == point:
            return
        self.points.append(point)

    def __len__(self):
        return len(self.points)


class Canvas:
    """Vector stroke store, rendered on demand.

    Strokes are kept as point lists rather than baked into a raster buffer so
    that erase, undo and the OCR export (which needs a clean crop at a
    different scale) all stay possible after the fact.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.strokes = []
        self._active = None
        self.color_index = 0
        self.thickness = 6

    @property
    def color(self):
        return COLORS[self.color_index][1]

    @property
    def color_name(self):
        return COLORS[self.color_index][0]

    def cycle_color(self):
        self.color_index = (self.color_index + 1) % len(COLORS)

    # -- drawing ---------------------------------------------------------

    def pen_down(self, point):
        if self._active is None:
            self._active = Stroke(self.color, self.thickness)
            self.strokes.append(self._active)
        self._active.add(point)

    def pen_up(self):
        # Drop stray one-point strokes left by a brief gesture flicker.
        if self._active is not None and len(self._active) < 2:
            self.strokes.remove(self._active)
        self._active = None

    def erase_near(self, point, radius=40):
        """Delete whole strokes intersecting the cursor.

        Point-level erase would split a stroke into fragments that render with
        visible seams; at air-writing scale, dropping the whole stroke is both
        simpler and closer to what people expect.
        """
        self.pen_up()
        px, py = point
        r2 = radius * radius
        kept = []
        for stroke in self.strokes:
            hit = any((x - px) ** 2 + (y - py) ** 2 <= r2 for x, y in stroke.points)
            if not hit:
                kept.append(stroke)
        removed = len(self.strokes) - len(kept)
        self.strokes = kept
        return removed

    def undo(self):
        self.pen_up()
        if self.strokes:
            self.strokes.pop()

    def clear(self):
        self._active = None
        self.strokes.clear()

    def is_empty(self):
        return not any(len(s) >= 2 for s in self.strokes)

    # -- rendering -------------------------------------------------------

    def draw_on(self, frame):
        """Composite strokes over the live camera frame."""
        for stroke in self.strokes:
            if len(stroke) < 2:
                continue
            pts = np.array(stroke.points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, stroke.color, stroke.thickness,
                          cv2.LINE_AA)
        return frame

    def bounding_box(self, padding=0):
        pts = [p for s in self.strokes if len(s) >= 2 for p in s.points]
        if not pts:
            return None
        xs, ys = zip(*pts)
        return (min(xs) - padding, min(ys) - padding,
                max(xs) + padding, max(ys) + padding)

    def render_for_ocr(self, target_height=128, padding=24, thickness=None):
        """Render strokes as black ink on white, cropped and scaled.

        OCR models are trained on tight, high-contrast, dark-on-light document
        crops. Feeding them a full-frame image with colored lines on black --
        which is what the on-screen canvas looks like -- fails badly. This
        rebuilds the same strokes in the form the model expects.
        """
        box = self.bounding_box(padding=padding)
        if box is None:
            return None

        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return None

        # Render at source scale first, then downscale once, so the line keeps
        # its antialiasing instead of aliasing at the target size.
        img = np.full((h, w), 255, dtype=np.uint8)
        ink = thickness or max(2, self.thickness)
        for stroke in self.strokes:
            if len(stroke) < 2:
                continue
            pts = np.array(stroke.points, dtype=np.int32) - np.array([x0, y0])
            cv2.polylines(img, [pts], False, 0, ink, cv2.LINE_AA)

        scale = target_height / h
        new_size = (max(1, int(round(w * scale))), target_height)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(img, new_size, interpolation=interp)

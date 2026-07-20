"""Gesture classification from MediaPipe hand landmarks."""

from collections import deque

# MediaPipe Hands landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
FINGER_TIPS = (4, 8, 12, 16, 20)
FINGER_PIPS = (3, 6, 10, 14, 18)

DRAW = "draw"
ERASE = "erase"
IDLE = "idle"
CLEAR = "clear"
NONE = "none"


def fingers_up(landmarks, handedness="Right"):
    """Return [thumb, index, middle, ring, pinky] booleans.

    For the four fingers, "up" means the tip is above (smaller y than) the PIP
    joint. The thumb folds sideways rather than down, so it needs an x-axis
    test whose direction depends on which hand it is.
    """
    up = []

    # Thumb: compare tip against the joint below it along x.
    if handedness == "Right":
        up.append(landmarks[THUMB_TIP].x < landmarks[THUMB_TIP - 1].x)
    else:
        up.append(landmarks[THUMB_TIP].x > landmarks[THUMB_TIP - 1].x)

    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        up.append(landmarks[tip].y < landmarks[pip].y)

    return up


def classify(landmarks, handedness="Right"):
    """Map a hand pose to an action."""
    up = fingers_up(landmarks, handedness)
    _, index, middle, ring, pinky = up

    if index and not middle and not ring and not pinky:
        return DRAW
    if index and middle and not ring and not pinky:
        return ERASE
    if not any(up):
        return CLEAR
    if all(up):
        return IDLE
    return IDLE


class GestureDebouncer:
    """Require a gesture to hold for several frames before accepting it.

    MediaPipe occasionally misreads a pose for one or two frames. Without this,
    a single bad frame lifts the pen mid-letter and shatters the stroke into
    fragments that OCR cannot read.
    """

    def __init__(self, window=4):
        self.window = window
        self._history = deque(maxlen=window)
        self.current = NONE

    def update(self, gesture):
        self._history.append(gesture)
        if len(self._history) == self.window and len(set(self._history)) == 1:
            self.current = gesture
        return self.current

    def reset(self):
        self._history.clear()
        self.current = NONE

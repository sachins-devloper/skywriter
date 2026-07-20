"""MediaPipe Hands wrapper producing a smoothed fingertip pointer.

Built on the MediaPipe Tasks API. The older `mp.solutions.hands` interface --
what most air-writing examples use -- was removed in mediapipe 0.10.3x, and
Tasks needs an explicit model file rather than bundling one.
"""

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .filters import PointFilter
from .gestures import INDEX_TIP, classify, GestureDebouncer, NONE

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

# Landmark index pairs forming the hand skeleton, for --debug rendering.
# Tasks does not ship the drawing helpers the old solutions API had.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]


class HandTracker:
    def __init__(self, model_path=None, detection_confidence=0.7,
                 tracking_confidence=0.6, smoothing=1.0, responsiveness=0.007,
                 debounce=4):
        model_path = Path(model_path or DEFAULT_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}.\n"
                f"Download it with:\n"
                f"  curl -L -o {model_path} --create-dirs {MODEL_URL}"
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._filter = PointFilter(min_cutoff=smoothing, beta=responsiveness)
        self._debouncer = GestureDebouncer(window=debounce)
        # VIDEO mode requires strictly increasing integer millisecond stamps,
        # so we count frames instead of trusting the wall clock.
        self._frame_index = 0

    def process(self, frame, timestamp):
        """Return (gesture, point, landmarks).

        point is the smoothed index fingertip in frame pixel coordinates, or
        None when no hand is visible. timestamp is seconds, used only by the
        smoothing filter.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_index += 1
        result = self._landmarker.detect_for_video(image, self._frame_index * 33)

        if not result.hand_landmarks:
            self._debouncer.reset()
            return NONE, None, None

        landmarks = result.hand_landmarks[0]

        handedness = "Right"
        if result.handedness:
            handedness = result.handedness[0][0].category_name
        # The frame is mirrored for the user, so MediaPipe's label is inverted
        # relative to the hand they are actually holding up.
        handedness = "Left" if handedness == "Right" else "Right"

        gesture = self._debouncer.update(classify(landmarks, handedness))

        tip = landmarks[INDEX_TIP]
        x, y = self._filter(tip.x * w, tip.y * h, timestamp)
        return gesture, (int(round(x)), int(round(y))), landmarks

    def draw_skeleton(self, frame, landmarks):
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame, (x, y), 3, (0, 140, 255), -1, cv2.LINE_AA)

    def close(self):
        self._landmarker.close()

"""Facial sentiment estimation from MediaPipe face blendshapes.

MediaPipe's FaceLandmarker emits 52 ARKit-style blendshape weights (how open
the jaw is, how raised each brow is, and so on). This module scores those
weights into coarse expression labels.

A caveat worth stating plainly: this is a **heuristic over muscle activations**,
not a trained emotion classifier. It reads what the face is *doing*, and infers
sentiment from that. Those are not the same thing -- a polite smile and genuine
delight produce near-identical blendshapes, and expressions vary across people
and cultures. Treat the output as "this face is smiling", not "this person is
happy".

The scoring is a pure function of a blendshape dict, so it is testable without
a camera or MediaPipe.
"""

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Each expression scores as a weighted sum of blendshapes. Weights are the
# relative contribution of each signal; they were set by hand, so treat them
# as a starting point rather than a calibrated model.
EXPRESSIONS = {
    "happy": {
        "mouthSmileLeft": 1.0, "mouthSmileRight": 1.0,
        "cheekSquintLeft": 0.5, "cheekSquintRight": 0.5,
    },
    "sad": {
        "mouthFrownLeft": 1.0, "mouthFrownRight": 1.0,
        "browInnerUp": 0.6, "mouthLowerDownLeft": 0.2,
        "mouthLowerDownRight": 0.2,
    },
    "surprised": {
        "jawOpen": 1.0, "eyeWideLeft": 0.7, "eyeWideRight": 0.7,
        "browOuterUpLeft": 0.5, "browOuterUpRight": 0.5,
    },
    "angry": {
        "browDownLeft": 1.0, "browDownRight": 1.0,
        "noseSneerLeft": 0.4, "noseSneerRight": 0.4,
        "mouthPressLeft": 0.3, "mouthPressRight": 0.3,
    },
}

# Below this, the face is not doing anything distinctive enough to call.
NEUTRAL_THRESHOLD = 0.18

EMOJI = {"happy": ":)", "sad": ":(", "surprised": ":O", "angry": ">:(",
         "neutral": ":|"}

COLORS = {  # BGR
    "happy": (80, 220, 80), "sad": (220, 140, 60),
    "surprised": (60, 220, 240), "angry": (70, 70, 235),
    "neutral": (190, 190, 190),
}


def score_expressions(blendshapes):
    """Score every expression from a {name: weight} dict. Weights normalised."""
    scores = {}
    for name, weights in EXPRESSIONS.items():
        total = sum(blendshapes.get(k, 0.0) * w for k, w in weights.items())
        scores[name] = total / sum(weights.values())
    return scores


def classify(blendshapes):
    """Return (label, confidence) for a blendshape dict."""
    if not blendshapes:
        return "neutral", 0.0
    scores = score_expressions(blendshapes)
    label = max(scores, key=scores.get)
    confidence = scores[label]
    if confidence < NEUTRAL_THRESHOLD:
        return "neutral", 1.0 - confidence
    return label, min(1.0, confidence)


class FaceSentiment:
    def __init__(self, model_path=None, detection_confidence=0.5, smoothing=0.6):
        model_path = Path(model_path or DEFAULT_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}.\n"
                f"Download it with:\n"
                f"  curl -L -o {model_path} --create-dirs {MODEL_URL}"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=detection_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._frame_index = 0
        # Blendshapes are noisy frame to frame; without smoothing the label
        # flickers between two expressions several times a second.
        self._smoothing = smoothing
        self._smoothed = {}

    def process(self, frame):
        """Return (label, confidence, blendshapes, landmarks)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_index += 1
        result = self._landmarker.detect_for_video(image, self._frame_index * 33)

        if not result.face_blendshapes:
            self._smoothed = {}
            return None, 0.0, {}, None

        raw = {c.category_name: c.score for c in result.face_blendshapes[0]}

        a = self._smoothing
        self._smoothed = {
            k: v if k not in self._smoothed
            else a * v + (1 - a) * self._smoothed[k]
            for k, v in raw.items()
        }

        label, confidence = classify(self._smoothed)
        landmarks = result.face_landmarks[0] if result.face_landmarks else None
        return label, confidence, self._smoothed, landmarks

    def close(self):
        self._landmarker.close()


def draw_face_mesh(frame, landmarks, color=(90, 90, 90)):
    """Sparse landmark dots. The full 478-point mesh obscures the face."""
    h, w = frame.shape[:2]
    for lm in landmarks[::6]:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, color, -1)

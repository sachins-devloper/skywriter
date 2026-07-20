"""Pluggable handwriting recognition backends.

Every backend is imported lazily. The heavy ones (TrOCR pulls torch and a
~1.3GB checkpoint) are optional, so the capture loop stays runnable with
nothing but OpenCV installed -- recognition then degrades to saving the crop
for offline processing rather than crashing the app.
"""

import cv2


class OCRBackend:
    name = "base"

    def available(self):
        raise NotImplementedError

    def recognize(self, image):
        """image: grayscale uint8, black ink on white. Returns text."""
        raise NotImplementedError


class NullOCR(OCRBackend):
    """Fallback when no recognizer is installed."""

    name = "none"

    def available(self):
        return True

    def recognize(self, image):
        return ""


class TesseractOCR(OCRBackend):
    name = "tesseract"

    def __init__(self, whitelist=None):
        # Air-written text is one line of large glyphs. PSM 7 (single text
        # line) beats the default layout analysis, which looks for paragraphs.
        self.config = "--psm 7"
        if whitelist:
            self.config += f" -c tessedit_char_whitelist={whitelist}"

    def available(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def recognize(self, image):
        import pytesseract
        return pytesseract.image_to_string(image, config=self.config).strip()


class TrOCRBackend(OCRBackend):
    name = "trocr"

    def __init__(self, model="microsoft/trocr-base-handwritten"):
        self.model_id = model
        self._proc = None
        self._model = None

    def available(self):
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        if self._model is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self._proc = TrOCRProcessor.from_pretrained(self.model_id)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
            self._model.eval()

    def recognize(self, image):
        import torch
        self._load()
        # TrOCR's processor expects 3-channel RGB.
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pixels = self._proc(images=rgb, return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = self._model.generate(pixels, max_new_tokens=64)
        return self._proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


class PaddleBackend(OCRBackend):
    name = "paddle"

    def __init__(self, lang="en"):
        self.lang = lang
        self._ocr = None

    def available(self):
        try:
            import paddleocr  # noqa: F401
            return True
        except ImportError:
            return False

    def recognize(self, image):
        from paddleocr import PaddleOCR
        if self._ocr is None:
            self._ocr = PaddleOCR(use_angle_cls=False, lang=self.lang,
                                  show_log=False)
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        result = self._ocr.ocr(rgb, cls=False)
        if not result or not result[0]:
            return ""
        return " ".join(line[1][0] for line in result[0])


BACKENDS = {
    "trocr": TrOCRBackend,
    "paddle": PaddleBackend,
    "tesseract": TesseractOCR,
    "none": NullOCR,
}


def load(name="auto"):
    """Return a usable backend, falling back if the requested one is missing."""
    if name != "auto":
        backend = BACKENDS[name]()
        if not backend.available():
            raise RuntimeError(
                f"OCR backend '{name}' is not installed. "
                f"See README for install instructions."
            )
        return backend

    for candidate in ("trocr", "paddle", "tesseract"):
        backend = BACKENDS[candidate]()
        if backend.available():
            return backend
    return NullOCR()

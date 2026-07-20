"""Clickable intro screen for choosing a mode."""

import cv2
import numpy as np

WIDTH, HEIGHT = 900, 600
WINDOW = "Air Studio"

BG = (28, 26, 24)
CARD = (46, 43, 40)
CARD_HOVER = (66, 62, 58)
ACCENT = (235, 200, 80)
TEXT = (240, 240, 240)
MUTED = (150, 148, 145)


class Card:
    def __init__(self, key, title, subtitle, lines, rect, accent):
        self.key = key
        self.title = title
        self.subtitle = subtitle
        self.lines = lines
        self.rect = rect          # (x, y, w, h)
        self.accent = accent

    def contains(self, x, y):
        cx, cy, cw, ch = self.rect
        return cx <= x <= cx + cw and cy <= y <= cy + ch


CARDS = [
    Card("airwrite", "Air Writing", "write with your fingertip",
         ["track the index fingertip",
          "rebuild strokes on a canvas",
          "read them back as text"],
         (60, 200, 380, 300), (235, 200, 80)),
    Card("sentiment", "Face Sentiment", "read expressions live",
         ["52 facial blendshapes",
          "scored into expressions",
          "happy / sad / surprised / angry"],
         (460, 200, 380, 300), (120, 200, 255)),
]


def _text(img, s, org, scale=0.6, color=TEXT, weight=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight,
                cv2.LINE_AA)


def _render(hover, preview=None):
    img = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)

    # A dim live preview doubles as a camera check -- if this is black, the
    # problem is the camera, before any mode is even entered.
    if preview is not None:
        ph = 150
        pw = int(preview.shape[1] * ph / preview.shape[0])
        thumb = cv2.resize(preview, (pw, ph), interpolation=cv2.INTER_AREA)
        x = WIDTH - pw - 30
        roi = img[30:30 + ph, x:x + pw]
        cv2.addWeighted(thumb, 0.35, roi, 0.65, 0, roi)
        cv2.rectangle(img, (x, 30), (x + pw, 30 + ph), (70, 70, 70), 1)
        _text(img, "camera", (x + 6, 30 + ph - 8), 0.4, MUTED)

    _text(img, "AIR STUDIO", (60, 80), 1.5, TEXT, 3)
    _text(img, "gesture and expression tools, running on one webcam",
          (62, 115), 0.6, MUTED)
    cv2.line(img, (60, 145), (WIDTH - 60, 145), (60, 58, 55), 1)

    for i, card in enumerate(CARDS):
        x, y, w, h = card.rect
        active = hover is card
        cv2.rectangle(img, (x, y), (x + w, y + h),
                      CARD_HOVER if active else CARD, -1)
        # Accent edge thickens on hover, so the target is obvious at a glance.
        cv2.rectangle(img, (x, y), (x + w, y + h), card.accent,
                      3 if active else 1)

        _text(img, f"{i + 1}", (x + 24, y + 52), 1.0, card.accent, 2)
        _text(img, card.title, (x + 62, y + 50), 0.95, TEXT, 2)
        _text(img, card.subtitle, (x + 64, y + 78), 0.5, MUTED)

        ly = y + 120
        for line in card.lines:
            cv2.circle(img, (x + 30, ly - 5), 3, card.accent, -1)
            _text(img, line, (x + 44, ly), 0.5, TEXT)
            ly += 30

        label = "click to start" if active else f"click  or press {i + 1}"
        _text(img, label, (x + 24, y + h - 24), 0.5,
              card.accent if active else MUTED)

    _text(img, "press q to quit  |  q inside a mode returns here",
          (60, HEIGHT - 28), 0.5, MUTED)
    return img


def show(get_frame=None):
    """Display the menu until a choice is made.

    get_frame: optional callable returning a camera frame for the preview.
    Returns "airwrite", "sentiment", or None if the user quit.
    """
    state = {"hover": None, "chosen": None}

    def on_mouse(event, x, y, flags, _):
        card = next((c for c in CARDS if c.contains(x, y)), None)
        state["hover"] = card
        if event == cv2.EVENT_LBUTTONDOWN and card is not None:
            state["chosen"] = card.key

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while True:
            preview = get_frame() if get_frame else None
            cv2.imshow(WINDOW, _render(state["hover"], preview))

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key == ord("1"):
                return CARDS[0].key
            if key == ord("2"):
                return CARDS[1].key
            if state["chosen"]:
                return state["chosen"]

            # Closing the window with the X button should quit, not spin.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyWindow(WINDOW)

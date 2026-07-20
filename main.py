"""Air-writing capture loop.

Write in the air with your index finger; the strokes are rebuilt on a virtual
canvas and handed to an OCR model on demand.

    python main.py                 # auto-select an OCR backend
    python main.py --ocr tesseract
    python main.py --no-mirror     # if your camera is not already mirrored
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from airwrite import ocr as ocr_module
from airwrite.canvas import Canvas
from airwrite.gestures import CLEAR, DRAW, ERASE
from airwrite.tracker import HandTracker

# Holding a fist clears the board. It fires on a timer rather than instantly
# because a fist is also what a hand looks like mid-way to any other pose.
CLEAR_HOLD_SECONDS = 1.0

HELP_LINES = [
    "index finger      draw",
    "index + middle    erase",
    "hold fist 1s      clear",
    "[r] recognise  [u] undo  [c] color  [s] save  [h] help  [q] quit",
]


def build_parser():
    """Shared by main.py and the app.py launcher, so flags stay identical."""
    p = argparse.ArgumentParser(description="Air-writing recognition")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--ocr", default="auto",
                   choices=list(ocr_module.BACKENDS) + ["auto"])
    p.add_argument("--no-mirror", action="store_true",
                   help="do not flip the frame horizontally")
    p.add_argument("--smoothing", type=float, default=1.0,
                   help="lower is smoother but laggier (One Euro min_cutoff)")
    p.add_argument("--outdir", default="captures")
    p.add_argument("--debug", action="store_true",
                   help="draw the hand skeleton")
    p.add_argument("--max-display-width", type=int, default=1100,
                   help="shrink the window if the camera returns a large frame")
    p.add_argument("--enhance", action="store_true",
                   help="boost local contrast; helps in dark or backlit rooms")
    p.add_argument("--detection-confidence", type=float, default=0.5,
                   help="lower it if your hand or face is not being picked up")
    return p


def parse_args():
    return build_parser().parse_args()


def enhance(frame):
    """Lift a dark, backlit frame so the palm detector can see the hand.

    A global brightness gain blows out the bright background without helping
    the shadowed foreground. CLAHE equalises locally on the luminance channel
    only, which lifts the hand while leaving a sunlit window alone -- and
    leaves color untouched, so the detector sees normal skin tones.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Gamma first: CLAHE equalises local contrast, but a frame that is dark
    # everywhere has little local contrast to work with, so on its own it
    # barely moves. Gamma lifts the whole shadow range into a usable band,
    # then CLAHE sharpens the hand against its background.
    gamma = np.array([((i / 255.0) ** 0.55) * 255 for i in range(256)],
                     dtype=np.uint8)
    l = cv2.LUT(l, gamma)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def draw_hud(frame, lines, origin=(12, 28), scale=0.6, color=(230, 230, 230)):
    x, y = origin
    for line in lines:
        # Dark pass underneath keeps the text readable over a bright frame.
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)
        y += int(30 * scale / 0.6)


def run(args, cap, window="Air Writing"):
    """Run the air-writing loop on an already-open camera.

    The launcher in app.py opens the camera once and hands it to whichever
    mode the user picks; re-acquiring a webcam per mode switch is slow on
    Windows and sometimes fails outright.
    """
    backend = ocr_module.load(args.ocr)
    if backend.name == "none":
        print("No OCR backend installed - strokes will be saved but not read.")
        print("Install one:  pip install pytesseract   (plus the Tesseract binary)")
    else:
        print(f"OCR backend: {backend.name}")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera returned no frames")
    h, w = frame.shape[:2]

    tracker = HandTracker(smoothing=args.smoothing,
                          detection_confidence=args.detection_confidence)
    canvas = Canvas(w, h)
    outdir = Path(args.outdir)

    recognized = ""
    status = ""
    status_until = 0.0
    fist_since = None
    show_help = True

    def flash(message, seconds=2.0):
        nonlocal status, status_until
        status = message
        status_until = time.time() + seconds

    def recognize():
        nonlocal recognized
        if canvas.is_empty():
            flash("nothing written")
            return
        image = canvas.render_for_ocr()
        if image is None:
            flash("nothing written")
            return
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = outdir / f"airwrite-{stamp}.png"
        cv2.imwrite(str(path), image)

        if backend.name == "none":
            recognized = ""
            flash(f"saved {path.name} (no OCR backend)", 3.0)
            return

        flash("recognising...", 30.0)
        try:
            text = backend.recognize(image)
        except Exception as exc:  # a missing model file should not kill the app
            recognized = ""
            flash(f"OCR failed: {exc}", 4.0)
            return
        recognized = text
        flash(f"read: {text!r}" if text else "OCR returned nothing", 3.0)
        print(f"{path.name} -> {text!r}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            if args.enhance:
                frame = enhance(frame)

            gesture, point, landmarks = tracker.process(frame, time.time())

            if gesture == DRAW and point is not None:
                canvas.pen_down(point)
            else:
                canvas.pen_up()

            if gesture == ERASE and point is not None:
                canvas.erase_near(point)

            # Fist has to be held; a momentary fist is usually a transition.
            if gesture == CLEAR:
                now = time.time()
                fist_since = fist_since or now
                if now - fist_since >= CLEAR_HOLD_SECONDS:
                    canvas.clear()
                    recognized = ""
                    fist_since = None
                    flash("cleared")
            else:
                fist_since = None

            canvas.draw_on(frame)

            if args.debug and landmarks is not None:
                tracker.draw_skeleton(frame, landmarks)

            if point is not None:
                # Cursor: filled while drawing, hollow while hovering.
                filled = -1 if gesture == DRAW else 2
                cv2.circle(frame, point, 9, canvas.color, filled, cv2.LINE_AA)
                if gesture == ERASE:
                    cv2.circle(frame, point, 40, (120, 120, 120), 2, cv2.LINE_AA)

            if fist_since is not None:
                progress = min(1.0, (time.time() - fist_since) / CLEAR_HOLD_SECONDS)
                cv2.rectangle(frame, (12, h - 24),
                              (12 + int(200 * progress), h - 12),
                              (60, 60, 220), -1)

            # Without this, an undetected hand is silent and looks identical to
            # a bug in the drawing code.
            if point is None:
                cv2.rectangle(frame, (0, 0), (w, 44), (0, 0, 130), -1)
                cv2.putText(frame, "NO HAND DETECTED - more light, or try --enhance",
                            (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

            hud = [f"{gesture}  |  {canvas.color_name}  |  "
                   f"{len(canvas.strokes)} strokes  |  ocr:{backend.name}"]
            if show_help:
                hud += HELP_LINES
            if time.time() < status_until:
                hud.append(status)
            draw_hud(frame, hud, origin=(12, 72 if point is None else 28))

            if recognized:
                cv2.putText(frame, recognized, (12, h - 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6,
                            cv2.LINE_AA)
                cv2.putText(frame, recognized, (12, h - 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
                            cv2.LINE_AA)

            # Cameras often ignore the requested capture size, and a frame
            # wider than the screen pushes the HUD off the left edge.
            if w > args.max_display_width:
                scale = args.max_display_width / w
                display = cv2.resize(frame, (args.max_display_width,
                                             int(round(h * scale))),
                                     interpolation=cv2.INTER_AREA)
            else:
                display = frame
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                recognize()
            elif key == ord("u"):
                canvas.undo()
            elif key == ord("c"):
                canvas.cycle_color()
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("s"):
                image = canvas.render_for_ocr()
                if image is None:
                    flash("nothing written")
                else:
                    outdir.mkdir(parents=True, exist_ok=True)
                    path = outdir / f"airwrite-{time.strftime('%Y%m%d-%H%M%S')}.png"
                    cv2.imwrite(str(path), image)
                    flash(f"saved {path.name}")
    finally:
        # The camera belongs to the caller; only tear down what we created.
        tracker.close()
        cv2.destroyWindow(window)


def main():
    from app import open_camera

    args = parse_args()
    cap = open_camera(args)
    try:
        run(args, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

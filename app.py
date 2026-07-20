"""Air Studio launcher.

Shows the intro screen, opens the camera once, and hands it to whichever mode
is chosen. Quitting a mode returns here rather than exiting, so switching
between features does not re-acquire the webcam -- which is slow on Windows
and occasionally fails outright.

    python app.py
    python app.py --mode sentiment     # skip the menu
"""

import cv2

import main as writing_mode
import sentiment as sentiment_mode
from airwrite import menu

MODES = {
    "airwrite": writing_mode.run,
    "sentiment": sentiment_mode.run,
}


def open_camera(args):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera opened but returned no frames")
    h, w = frame.shape[:2]
    if (w, h) != (args.width, args.height):
        print(f"Camera returned {w}x{h} (requested {args.width}x{args.height})")
    return cap


def main():
    parser = writing_mode.build_parser()
    parser.add_argument("--mode", choices=list(MODES),
                        help="skip the menu and start this mode directly")
    args = parser.parse_args()

    cap = open_camera(args)

    def preview():
        ok, frame = cap.read()
        return cv2.flip(frame, 1) if ok and not args.no_mirror else (frame if ok else None)

    try:
        if args.mode:
            MODES[args.mode](args, cap)
            return
        while True:
            choice = menu.show(get_frame=preview)
            if choice is None:
                break
            MODES[choice](args, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

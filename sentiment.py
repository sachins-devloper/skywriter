"""Face sentiment mode: read expressions from the camera and label them.

Sentiment here is inferred from MediaPipe blendshapes -- how open the jaw is,
how raised each brow is, how much each mouth corner pulls. See
airwrite/face.py for why that is a description of the face, not a claim about
how the person feels.
"""

import time

import cv2

from airwrite.face import COLORS, EMOJI, EXPRESSIONS, FaceSentiment, draw_face_mesh, score_expressions

BAR_WIDTH = 220


def draw_scores(frame, blendshapes, origin=(14, 120)):
    """Bar chart of every expression score, not just the winner.

    Showing only the label hides how close the call was; a 0.31 'happy' beside
    a 0.29 'surprised' is worth seeing rather than trusting blindly.
    """
    if not blendshapes:
        return
    x, y = origin
    for name, score in sorted(score_expressions(blendshapes).items(),
                              key=lambda kv: -kv[1]):
        color = COLORS.get(name, (200, 200, 200))
        cv2.rectangle(frame, (x, y - 12), (x + BAR_WIDTH, y + 4), (45, 45, 45), -1)
        cv2.rectangle(frame, (x, y - 12),
                      (x + int(BAR_WIDTH * min(1.0, score)), y + 4), color, -1)
        cv2.putText(frame, f"{name} {score:.2f}", (x + BAR_WIDTH + 10, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"{name} {score:.2f}", (x + BAR_WIDTH + 10, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        y += 26


def run(args, cap, window="Face Sentiment"):
    """Run the sentiment loop on an already-open camera."""
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera returned no frames")
    h, w = frame.shape[:2]

    detector = FaceSentiment(detection_confidence=args.detection_confidence)
    show_scores = True
    show_mesh = args.debug

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            if args.enhance:
                from main import enhance
                frame = enhance(frame)

            label, confidence, blendshapes, landmarks = detector.process(frame)

            if label is None:
                cv2.rectangle(frame, (0, 0), (w, 44), (0, 0, 130), -1)
                cv2.putText(frame, "NO FACE DETECTED - more light, or try --enhance",
                            (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
            else:
                if show_mesh and landmarks is not None:
                    draw_face_mesh(frame, landmarks)

                color = COLORS.get(label, (200, 200, 200))
                text = f"{EMOJI.get(label, '')}  {label.upper()}  {confidence:.0%}"
                cv2.putText(frame, text, (14, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (0, 0, 0), 7, cv2.LINE_AA)
                cv2.putText(frame, text, (14, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, color, 2, cv2.LINE_AA)

                if show_scores:
                    draw_scores(frame, blendshapes)

            cv2.putText(frame, "[b] bars  [m] mesh  [s] save  [q] menu",
                        (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, "[b] bars  [m] mesh  [s] save  [q] menu",
                        (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (225, 225, 225), 1, cv2.LINE_AA)

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
            elif key == ord("b"):
                show_scores = not show_scores
            elif key == ord("m"):
                show_mesh = not show_mesh
            elif key == ord("s"):
                from pathlib import Path
                outdir = Path(args.outdir)
                outdir.mkdir(parents=True, exist_ok=True)
                path = outdir / f"sentiment-{time.strftime('%Y%m%d-%H%M%S')}.png"
                cv2.imwrite(str(path), frame)
                print(f"saved {path}")
    finally:
        detector.close()
        cv2.destroyWindow(window)

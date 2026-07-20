# Air Writing

Write in the air with your index finger. The camera tracks the fingertip,
rebuilds the strokes on a virtual canvas, and hands them to an OCR model.

```
camera -> MediaPipe Hands -> index fingertip -> One Euro filter
       -> stroke builder -> canvas -> cropped image -> OCR -> text
```

## Setup

MediaPipe does not support Python 3.13+. Use 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# The Tasks API needs an explicit model file (~7MB, not bundled with the wheel)
curl -L -o models/hand_landmarker.task --create-dirs `
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

.\.venv\Scripts\python.exe main.py
```

The app runs without an OCR backend — it just saves the stroke crop instead of
reading it. Install one to get text:

| Backend | Install | Notes |
|---|---|---|
| Tesseract | `pip install pytesseract` + [the binary](https://github.com/UB-Mannheim/tesseract/wiki) | Fastest to set up. Weak on cursive. |
| TrOCR | `pip install transformers torch` | Best accuracy. ~1.3GB first run. |
| PaddleOCR | `pip install paddleocr paddlepaddle` | Good middle ground. |

Auto-selection prefers TrOCR, then Paddle, then Tesseract. Override with
`--ocr tesseract`.

## Controls

| Gesture | Action |
|---|---|
| Index finger only | Draw |
| Index + middle | Erase strokes near the cursor |
| Closed fist, held 1s | Clear the canvas |
| Open palm | Idle (pen up) |

| Key | Action |
|---|---|
| `r` | Recognize |
| `u` | Undo last stroke |
| `c` | Cycle color |
| `s` | Save the crop without running OCR |
| `h` | Toggle help |
| `q` / Esc | Quit |

Useful flags: `--camera 1`, `--debug` (draw the hand skeleton),
`--no-mirror`, `--smoothing 0.6` (lower is smoother but laggier).

### If nothing draws

The banner across the top tells you whether a hand is being detected at all.
If it says NO HAND DETECTED, the tracker never sees you and no amount of
gesturing will draw.

Almost always this is lighting. The palm detector needs the hand brighter than
its background, and a window behind you defeats that — the camera exposes for
the window and your hand goes to shadow.

```powershell
.\.venv\Scripts\python.exe main.py --enhance --debug
```

`--enhance` lifts shadows without blowing out the bright background;
`--debug` draws the skeleton so you can see exactly what is being tracked.
If it still fails, face a light source rather than having one behind you, and
try `--detection-confidence 0.3`.

Once the skeleton appears but strokes look wrong, the gesture readout in the
HUD is the thing to watch — it should say `draw` with only your index finger
extended.

## Layout

| File | Role |
|---|---|
| [main.py](main.py) | Capture loop, HUD, key handling |
| [airwrite/tracker.py](airwrite/tracker.py) | MediaPipe wrapper → smoothed pointer + gesture |
| [airwrite/gestures.py](airwrite/gestures.py) | Finger-state → action, with debouncing |
| [airwrite/filters.py](airwrite/filters.py) | One Euro filter |
| [airwrite/canvas.py](airwrite/canvas.py) | Stroke store, rendering, OCR export |
| [airwrite/ocr.py](airwrite/ocr.py) | Pluggable recognizer backends |

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

They cover gestures, filtering and the canvas with synthetic input — no camera
or MediaPipe needed.

## Three things that decide whether this works

**A note on the MediaPipe API.** Most air-writing examples online use
`mp.solutions.hands`. That interface was removed in mediapipe 0.10.3x — this
project uses the Tasks API (`vision.HandLandmarker`) instead, which is why the
model file is a separate download and why `--debug` draws its own skeleton
rather than calling `drawing_utils`.


**Smoothing.** Raw landmarks jitter several pixels per frame, which is enough
to make letters illegible. A plain EMA removes the jitter but lags behind fast
strokes and rounds off corners. The One Euro filter in
[filters.py](airwrite/filters.py) varies its cutoff with speed instead:
smooth when still, responsive when moving.

**Gesture debouncing.** MediaPipe misreads a pose for a frame or two now and
then. Acting on every frame lifts the pen mid-letter and shatters strokes into
fragments OCR can't read, so a gesture must hold for 4 frames
([gestures.py](airwrite/gestures.py)).

**The OCR export.** OCR models are trained on tight, dark-on-light document
crops. The on-screen canvas — colored lines on a camera frame — is nothing like
that. `render_for_ocr` rebuilds the strokes as black ink on white, cropped to
the bounding box and scaled to the model's expected height
([canvas.py](airwrite/canvas.py)).

## Not built yet

- **Multi-line writing.** Everything is treated as one line. Real notes need
  line segmentation before OCR.
- **Word segmentation.** No way to signal a space; a pause-based word break is
  the usual approach.
- **Web stack.** The FastAPI/React layer in the original plan. The tracking
  loop is the hard part and it lives here; a web version would stream frames
  over WebSocket to this same pipeline.
- **Trained air-writing model.** Off-the-shelf OCR is trained on pen-on-paper
  handwriting, which is neater and better-proportioned than anything written
  with a fingertip. A model fine-tuned on air-writing data is the real accuracy
  ceiling here.

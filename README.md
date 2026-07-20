# Air Writing

Write in the air with your index finger. A webcam tracks the fingertip, rebuilds
the strokes on a virtual canvas, and hands them to an OCR model to read back as
text.

> **Naming.** The package is currently `airwrite` and the folder is
> `handsignal`. `Inkless` is the proposed project name — see
> [Naming](#naming) before this spreads further.

---

## Contents

- [How it works](#how-it-works)
- [Face Sentiment](#face-sentiment)
- [Requirements](#requirements)
- [Install](#install)
- [Running it](#running-it)
- [Controls](#controls)
- [Command-line flags](#command-line-flags)
- [Architecture](#architecture)
- [The three problems that actually matter](#the-three-problems-that-actually-matter)
- [Tuning](#tuning)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)
- [Adding an OCR backend](#adding-an-ocr-backend)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Naming](#naming)

---

## How it works

```
webcam frame
     │
     ▼
MediaPipe HandLandmarker ──► 21 landmarks
     │
     ▼
landmark 8 (index fingertip) ──► One Euro filter ──► smoothed pointer
     │
     ▼
finger-state classifier ──► debouncer ──► draw / erase / clear / idle
     │
     ▼
stroke builder ──► vector canvas
     │
     ▼
render_for_ocr: black ink on white, cropped, scaled
     │
     ▼
OCR backend ──► recognised text
```

The canvas holds **vectors, not pixels**. Strokes stay as point lists so that
undo, erase and the OCR export — which needs a clean crop at a different scale
and polarity than the screen — all remain possible after the fact.

---

## Requirements

| | |
|---|---|
| Python | **3.12** — MediaPipe has no 3.13+ wheels |
| Camera | Any webcam OpenCV can open |
| OS | Developed on Windows 11; nothing platform-specific except the DirectShow camera backend |
| Disk | ~400MB for MediaPipe + OpenCV, plus 7MB for the hand model |

---

## Install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Models

The MediaPipe Tasks API does not bundle models — download both once:

```powershell
curl -L -o models/hand_landmarker.task --create-dirs `
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

curl -L -o models/face_landmarker.task --create-dirs `
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

The hand model is ~7MB, the face model ~4MB.

The app fails with the exact download command if the file is missing, so you
cannot get this wrong silently.

### OCR backend (optional)

The capture loop runs without one — recognition saves the stroke crop to
`captures/` instead of reading it. Install one to get text:

| Backend | Install | Trade-off |
|---|---|---|
| **Tesseract** | `pip install pytesseract` + [Windows binary](https://github.com/UB-Mannheim/tesseract/wiki) | Fastest setup, lightest. Weak on cursive and uneven letters. |
| **TrOCR** | `pip install transformers torch` | Best accuracy on handwriting. ~1.3GB download on first run, slowest per call. |
| **PaddleOCR** | `pip install paddleocr paddlepaddle` | Middle ground on both accuracy and weight. |

`--ocr auto` (the default) prefers TrOCR, then PaddleOCR, then Tesseract, then
falls back to saving crops. Force one with `--ocr tesseract`.

---

## Running it

```powershell
.\.venv\Scripts\python.exe app.py
```

This opens **Air Studio**, an intro screen listing the two features. Click a
card, or press `1` / `2`. Pressing `q` inside a mode returns to the menu rather
than exiting, and the camera is opened once and shared — switching modes does
not re-acquire the webcam.

The menu also shows a dim live camera thumbnail, which doubles as a check: if
that is black, the problem is the camera, before any mode is even entered.

| Mode | What it does |
|---|---|
| **Air Writing** | Track the fingertip, rebuild strokes, read them as text |
| **Face Sentiment** | Score facial blendshapes into expression labels |

Skip the menu with `--mode airwrite` or `--mode sentiment`. In a dark or
backlit room, start with:

```powershell
.\.venv\Scripts\python.exe app.py --enhance --debug
```

A window opens with your camera feed mirrored. The HUD line at the top shows
the detected gesture, active color, stroke count and OCR backend. A red banner
appears whenever no hand is detected.

**First run checklist:** hold up one index finger, confirm the HUD reads
`draw`, and write a single large capital letter. Press `r`. Even without an OCR
backend you get a PNG in `captures/` — if the letter is legible in that file,
tracking works and everything downstream is just model choice.

---

## Controls

### Gestures

| Gesture | Action |
|---|---|
| Index finger only | Draw |
| Index + middle | Erase strokes near the cursor |
| Closed fist, held 1s | Clear the canvas |
| Open palm | Idle — pen up |

Clear is on a **hold timer**, not instant, because a fist is also what a hand
passes through on the way to any other pose. A progress bar fills at the bottom
left while it counts down.

### Keys

| Key | Action |
|---|---|
| `r` | Recognize — export the crop and run OCR |
| `u` | Undo the last stroke |
| `c` | Cycle color (cyan → green → magenta → yellow) |
| `s` | Save the crop without running OCR |
| `h` | Toggle the help overlay |
| `q` / `Esc` | Quit |

---

## Face Sentiment

Reads MediaPipe's 52 ARKit-style blendshape weights — how open the jaw is, how
raised each brow is, how much each mouth corner pulls — and scores them into
**happy / sad / surprised / angry / neutral**. A bar chart shows every score,
not just the winner, because a 0.31 `happy` next to a 0.29 `surprised` is worth
seeing rather than trusting.

| Key | Action |
|---|---|
| `b` | Toggle the score bars |
| `m` | Toggle the face mesh |
| `s` | Save a screenshot |
| `q` | Back to the menu |

### What this is and is not

This is a **heuristic over muscle activations, not a trained emotion
classifier.** It reads what a face is *doing* and infers sentiment from that,
and those are not the same thing:

- A polite smile and genuine delight produce near-identical blendshapes.
- Expression-to-emotion mapping varies across people and cultures; the
  scientific consensus is that it is not universal.
- The weights in `EXPRESSIONS` ([airwrite/face.py](airwrite/face.py)) were set
  by hand, not fitted to labelled data.

Read the output as *"this face is smiling"*, not *"this person is happy"*. It
is not suitable for anything consequential — assessment, screening, monitoring
— and would need a properly trained and validated model for that.

---

## Command-line flags

| Flag | Default | Purpose |
|---|---|---|
| `--camera` | `0` | Camera index. Try `1`, `2` if the wrong device opens. |
| `--width` / `--height` | `1280` / `720` | Requested capture size. Cameras may ignore it. |
| `--ocr` | `auto` | `trocr`, `paddle`, `tesseract`, `none`, `auto` |
| `--no-mirror` | off | Skip the horizontal flip, if your camera mirrors already |
| `--smoothing` | `1.0` | One Euro `min_cutoff`. Lower = smoother, laggier |
| `--detection-confidence` | `0.5` | Lower it if your hand is not picked up |
| `--enhance` | off | Lift shadows for dark or backlit rooms |
| `--max-display-width` | `1100` | Shrink the window if the camera returns a large frame |
| `--debug` | off | Draw the hand skeleton |
| `--outdir` | `captures` | Where crops are written |

---

## Architecture

| File | Role |
|---|---|
| [app.py](app.py) | Launcher — opens the camera, shows the menu, dispatches modes |
| [airwrite/menu.py](airwrite/menu.py) | Clickable intro screen |
| [main.py](main.py) | Air-writing loop, HUD, key handling, frame enhancement |
| [sentiment.py](sentiment.py) | Face sentiment loop and score bars |
| [airwrite/face.py](airwrite/face.py) | Blendshape → expression scoring |
| [airwrite/tracker.py](airwrite/tracker.py) | MediaPipe Tasks wrapper → smoothed pointer + gesture |
| [airwrite/gestures.py](airwrite/gestures.py) | Finger-state → action, plus debouncing |
| [airwrite/filters.py](airwrite/filters.py) | One Euro adaptive smoothing filter |
| [airwrite/canvas.py](airwrite/canvas.py) | Stroke store, screen rendering, OCR export |
| [airwrite/ocr.py](airwrite/ocr.py) | Pluggable recognizer backends |
| [tests/test_pipeline.py](tests/test_pipeline.py) | Air-writing tests — no camera needed |
| [tests/test_sentiment.py](tests/test_sentiment.py) | Expression-scoring tests |

### A note on the MediaPipe API

Nearly every air-writing tutorial online uses `mp.solutions.hands`. **That
interface was removed in mediapipe 0.10.3x** — on 0.10.35, `dir(mediapipe)` is
just `['Image', 'ImageFormat', 'tasks']`.

This project uses the Tasks API (`vision.HandLandmarker`) instead. Three
consequences worth knowing if you copy code from a tutorial:

- the model is a separate download rather than bundled
- `drawing_utils` is gone, so `--debug` renders its own skeleton from an
  explicit connection list in [tracker.py](airwrite/tracker.py)
- `RunningMode.VIDEO` demands strictly increasing integer millisecond
  timestamps, so the tracker counts frames rather than reading the wall clock

---

## The three problems that actually matter

Most of the difficulty in air writing is not the tracking — MediaPipe handles
that. It is these three, and getting any one wrong makes output illegible.

### 1. Jitter versus lag

Raw landmarks wobble several pixels per frame, which is enough to make letters
unreadable. A plain exponential moving average removes the jitter but lags
behind fast strokes and rounds off corners — so letters lose exactly the sharp
features OCR relies on.

[filters.py](airwrite/filters.py) uses a **One Euro filter**, which varies its
cutoff frequency with pointer speed: heavy smoothing when the finger is nearly
still (where jitter shows), light smoothing when it moves fast (where lag
shows).

### 2. Gesture flicker

MediaPipe misreads a pose for a frame or two now and then. Acting on every
frame lifts the pen mid-letter and shatters one stroke into fragments that no
OCR model can read.

[gestures.py](airwrite/gestures.py) requires a gesture to hold for **4
consecutive frames** before it takes effect. A single bad frame is absorbed.

### 3. The OCR export

OCR models are trained on tight, high-contrast, dark-on-light document crops.
The on-screen canvas is the opposite: colored antialiased lines over a live
camera frame, surrounded by hundreds of pixels of irrelevant background.

`render_for_ocr` in [canvas.py](airwrite/canvas.py) rebuilds the same strokes
as **black ink on white**, cropped to the ink's bounding box with padding, and
scaled to the model's expected height — rendering at source scale first so the
line keeps its antialiasing instead of aliasing at the target size.

---

## Tuning

| Symptom | Try |
|---|---|
| Strokes look furry or shaky | `--smoothing 0.6` |
| Pointer lags behind your finger | `--smoothing 1.6` |
| Strokes break into fragments mid-letter | Raise `debounce` in [tracker.py](airwrite/tracker.py) |
| Pen feels sticky to lift | Lower `debounce` |
| Erase takes too much | Lower `radius` in `Canvas.erase_near` |
| Lines too thin or thick for OCR | `Canvas.thickness` in [canvas.py](airwrite/canvas.py) |

---

## Troubleshooting

### Nothing draws

**Read the banner first.** `NO HAND DETECTED` means the tracker never sees your
hand and no amount of gesturing will draw — that is a detection problem, not a
drawing problem. If there is no banner but nothing appears, watch the gesture
readout in the HUD instead: it should read `draw` with only your index finger
extended.

### Hand is not detected

Almost always lighting. The palm detector needs the hand brighter than its
background, and **a window behind you defeats that** — the camera exposes for
the window and drops your hand into shadow.

In order:

1. `--enhance` — gamma lift plus CLAHE on the luminance channel. On a dark
   backlit test frame this raised shadows from 32 to 89 while leaving a
   blown-out window at ~250 instead of pushing it further. A plain brightness
   gain would wreck the background without helping the hand.
2. Face a light source rather than having one behind you.
3. `--detection-confidence 0.3`
4. `--debug` to confirm what is actually being tracked.

### The window is bigger than my screen

Cameras frequently ignore the requested capture size. `--max-display-width`
scales the *display* only; tracking still runs at full resolution.

### OCR returns nothing or garbage

Open the PNG in `captures/`. That image is exactly what the model sees.

- Letters illegible there → a tracking problem, not an OCR problem
- Legible but misread → write larger, use block capitals, and prefer TrOCR
- Multiple words → currently unsupported, see [Limitations](#limitations)

### `ModuleNotFoundError: No module named 'mediapipe'`

You are on Python 3.13+. There are no MediaPipe wheels for it — use the 3.12
venv.

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

31 tests covering gesture classification, debouncing, One Euro filtering and
the canvas, plus expression scoring — including OCR-export polarity,
cropping and scale. All run on
synthetic input, **no camera or MediaPipe required**, so they work in CI.

Not covered: live hand and face detection, which need a real camera.

---

## Adding an OCR backend

Subclass `OCRBackend` in [airwrite/ocr.py](airwrite/ocr.py) and register it:

```python
class MyBackend(OCRBackend):
    name = "mine"

    def available(self):
        try:
            import my_ocr_lib  # noqa: F401
            return True
        except ImportError:
            return False

    def recognize(self, image):
        # image: grayscale uint8, black ink on white
        return my_ocr_lib.read(image)

BACKENDS["mine"] = MyBackend
```

Import heavy dependencies **inside** the methods, not at module scope — that is
what keeps the app runnable with no backend installed.

---

## Limitations

- **Single line only.** Everything written is treated as one line of text.
- **No word spacing.** There is no way to signal a space between words.
- **One hand.** `num_hands=1`; a second hand in frame is ignored.
- **Stroke-level erase.** Erasing removes whole strokes, not partial ones.
- **No persistence.** Strokes are lost on exit; only exported crops survive.
- **Off-the-shelf OCR.** Every available backend is trained on pen-on-paper
  handwriting, which is far neater and better-proportioned than anything
  written with a fingertip in mid-air. This is the real accuracy ceiling.

---

## Roadmap

Roughly in order of value per unit of effort:

1. **Word segmentation** — a pause-based break, so multi-word phrases work.
2. **Line segmentation** — group strokes into lines before OCR, enabling notes
   rather than single words.
3. **Session persistence** — save and reload stroke sets; export to PDF.
4. **The web stack** — the FastAPI/React layer from the original plan. Worth
   noting the tracking loop is the hard part and it already exists here; a web
   version streams frames over WebSocket into this same pipeline.
5. **LLM hookup** — send recognized text to a model and render the answer.
6. **A fine-tuned air-writing model** — the only thing that lifts the accuracy
   ceiling meaningfully, and by far the most work.

---

## Naming

The folder is `handsignal` and the package is `airwrite`, which is already
inconsistent and worth resolving before it spreads.

`handsignal` is actively misleading — this is not sign-language or
gesture-command recognition. **`Inkless`** is the proposed replacement: it
names the idea (handwriting with no ink, pen or surface) rather than the
mechanism, and is free on PyPI. `AirQuill`, `AirCanvas` and `Ghostpen` are also
free. Avoid `airscript` and `skywriter` — both are taken.

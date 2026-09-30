# Hand Fruit Slash Game

A webcam-controlled Fruit Ninja-style game made with Python.

## Features

- Real hand controls the game.
- The full mirrored webcam frame stays visible behind the game.
- Only the index fingertip (MediaPipe landmark 8) controls the slash.
- Fingertip tracking is smoothed and shows a fading sword trail.
- Fruits fall from the top.
- Fast finger movement creates a slash.
- Cutting fruit gives points.
- Each fruit has an individual score and appearance.
- Bomb collision immediately ends the game.
- A special bonus fruit is worth 50 points.
- Combo count increases for consecutive fruit cuts.
- Difficulty increases over time.
- Fruit split, floating score, and particle effects.
- Press R to restart after game over.
- Press ESC to quit.

## Fruit values

| Fruit | Points |
|---|---:|
| Apple | 10 |
| Kiwi | 15 |
| Strawberry | 20 |
| Orange | 8 |
| Cherry | 25 |
| Watermelon | 5 |
| Pineapple | 30 |
| Banana | 12 |
| Bonus | 50 |

## Requirements

Python 3.12 and the existing `.venv312` environment are used. Dependency versions are pinned to the tested environment.

## Installation

Open VS Code terminal in this folder:

```bash
source .venv312/bin/activate
pip install -r requirements.txt
```

On the first run, the program downloads the MediaPipe Hand Landmarker model into:

```text
models/hand_landmarker.task
```

## Run

```bash
python main.py
```

Give macOS camera permission if requested.

If the camera does not open:

System Settings -> Privacy & Security -> Camera

Enable camera access for the application that is running Python/VS Code.

## Controls

- Move index finger quickly through a fruit = slice
- Hit bomb = game over
- R = restart after game over
- ESC = quit

## How the hand detection works

MediaPipe detects the hand from the webcam. Landmark 8 is the index fingertip.

The program converts the fingertip's normalized camera coordinates into the displayed full-frame camera coordinates. It does not crop or zoom into the hand.

It compares the previous fingertip position with the current position. If the movement is fast enough, the line between those positions is treated as a slash.

The slash line is checked against each fruit/bomb using point-to-line-segment distance.

This means the hand is not simply clicking objects; it actually behaves like a blade.

The camera requests 1280x720 and falls back to a supported resolution. If startup fails, the game displays camera-permission guidance. The hand landmark model downloads to `models/hand_landmarker.task` on first run.

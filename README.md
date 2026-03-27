
# ASTRALINK — Space Debris Detection and Analysis

ASTRALINK is a **single-file Streamlit app** for space-debris and orbit visualization, plus a second image-based debris-analysis demo. It can load TLEs from **CelesTrak** or from a bundled sample set, propagate objects with **Skyfield**, render a **3D Plotly globe**, scan for close approaches, and store catalog/analysis records in **SQLite**. It also includes an image-analysis path that tries **YOLO** first and falls back to **OpenCV contour detection**.

## Overview

The project is built around a single main app file: `app.py`. From there, the app:
- initializes the local SQLite database,
- loads TLE data,
- converts TLEs into Skyfield satellites,
- propagates objects to the current time,
- visualizes the objects on a 3D globe,
- scans for close approaches,
- and saves catalog or analysis records into the database.

The image-analysis section accepts one or two uploaded images, detects an object, estimates basic properties, and stores the result in the same database.

## Features

- Load live TLEs from CelesTrak or use bundled sample TLE text
- Parse TLEs into Skyfield `EarthSatellite` objects
- Propagate orbit positions to the current time
- Display a 3D Plotly globe with orbit traces and object markers
- Scan for close approaches over a selected time window
- Draw a connector and demo avoidance trajectory when a close approach is found
- Store debris/catalog records in SQLite
- Show database contents in the app
- Export catalog data to CSV
- Run an image-analysis demo for debris detection
- Estimate size, shape, material, and optional speed from uploaded images
- Use YOLO if available, otherwise fall back to OpenCV contour detection

## How It Works

1. `app.py` starts the Streamlit interface and initializes the SQLite database.
2. The sidebar lets the user choose between live TLEs from CelesTrak and bundled sample TLE text.
3. The TLE text is parsed into Skyfield satellites.
4. Each object is propagated to the current time to obtain position-related values.
5. The app displays the orbit and current object markers on a 3D globe.
6. A close-approach scan checks the selected time window and threshold.
7. Matching records are written to SQLite and shown in the interface.
8. The image-analysis panel accepts one or two images, runs detection, estimates properties, and saves the output to the same database.

## Tech Stack

Confirmed from the repository:
- **Streamlit**
- **Pandas**
- **NumPy**
- **Plotly**
- **Skyfield**
- **OpenCV (`opencv-python-headless`)**
- **SQLite**
- **requests**
- **Pillow / PIL**
- Standard library modules such as `pathlib`, `sqlite3`, `datetime`, `math`, `hashlib`, and `io`

Optional runtime path:
- **YOLO / Ultralytics** is attempted if available, but it is not required for the fallback path

## Project Structure

```text
ASTRALINK/
├── .devcontainer/
│   └── devcontainer.json
├── app.py
├── debris_demo.db
├── requirements.txt
└── yolov8n.pt
```

## Main Entry File

The main entry file is **`app.py`**.

## Data Flow

**TLE path**
```text
TLE text (live CelesTrak or bundled sample)
→ parse into Skyfield satellites
→ propagate to current time
→ derive position data
→ render in Plotly
→ scan close approaches
→ store and display records in SQLite
```

**Image-analysis path**
```text
uploaded image(s)
→ YOLO if available
→ OpenCV fallback if needed
→ estimate debris properties
→ annotate and display result
→ save analysis record in SQLite
```

## Files in the Repository

- **`app.py`** — the complete Streamlit application: UI, TLE handling, orbit propagation, image analysis, database access, Plotly rendering, and close-approach scanning.
- **`requirements.txt`** — runtime dependencies.
- **`.devcontainer/devcontainer.json`** — development container setup that launches the app.
- **`debris_demo.db`** — bundled SQLite database file.
- **`yolov8n.pt`** — bundled YOLO weight file used by the optional YOLO-based path.

## Limitations

- The project is essentially a **single large script**; it is not split into separate modules.
- `requirements.txt` does not fully describe every import used at runtime.
- The image-analysis path includes a fallback and heuristic estimation logic.
- The project ships with a SQLite database file and a YOLO weight file in the repo root.
- The close-approach scan is a demo-style sampled scan rather than a more advanced conjunction analysis system.
- Some outputs are approximations intended for the demo and visualization workflow.

## What Can Be Improved

- Split `app.py` into smaller modules for TLE handling, orbit logic, image analysis, database, plotting, and UI.
- Make dependency listing match the actual imports.
- Generate the SQLite database on first run instead of storing it in the repo root.
- Add tests for TLE parsing, close-approach scanning, and database writes.
- Refine the image-analysis path so the fallback and saved outputs are clearer and more consistent.
- Clean up any incomplete or unused UI paths.
- Improve documentation and usage instructions once the codebase is more modular.

## Future Scope

Possible next steps that stay close to the current codebase:
- better structure and separation of concerns,
- stronger testing,
- a more explicit database schema and data flow,
- improved image-analysis accuracy,
- and more polished visualization and alert handling.

## Setup

```bash
git clone <your-repo-link>
cd ASTRALINK
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app can work with live CelesTrak TLE data or bundled sample TLE text.
- YOLO is optional; the fallback path uses OpenCV contour detection.
- The repository currently centers on a single working app file rather than a multi-module architecture.

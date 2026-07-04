# 🚦 Road Traffic Analyzer

Drop a street video into the page and get back an **annotated video** that
detects, tracks, and counts **cars, buses, trucks, motorcycles, and people**
using **YOLOv8** — along with live **traffic-density** readouts and a set of
analytics charts.

![Built with YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

## What it does

- **Detects** cars, buses, trucks, motorcycles, bicycles, and people.
- **Tracks** every object with a stable ID across frames (ByteTrack).
- **Counts** each one as it crosses a virtual line — once, by class and direction.
- **Measures density** in real time and labels it Low / Medium / High.
- **Returns an annotated video** with boxes, motion trails, the counting line, and a live dashboard.
- **Generates charts**: totals by type, composition, cumulative crossings, and density over time.

---

## Try it locally

```bash
pip install -r requirements.txt
python app.py
```


---


## All in One

`Road_Traffic_Analyzer.ipynb` is a Jupyter notebook that contains everything needed within it. It can stand alone without requiring any additional files or dependencies.

---

## Controls

| Control | What it does |
|---|---|
| **Model** | `yolov8n` (fast) or `yolov8s` (more accurate, slower) |
| **Detection confidence** | Higher = fewer, more certain detections |
| **Counting line position** | Where the line sits, top → bottom of the frame |
| **Seconds to analyze** | Caps processing time on the free CPU tier |

---

## How it works

1. **Detect** — YOLOv8 finds objects each frame and keeps only traffic classes.
2. **Track** — ByteTrack gives each object a persistent ID.
3. **Count** — a virtual line is drawn across the road; using the sign of a 2D
   cross product we know which side each object is on, and when a tracked object
   flips sides (within the line segment) it's counted once, by class and direction.
4. **Density** — the number of vehicles in frame is smoothed and mapped to
   Low / Medium / High.
5. **Report** — per-frame data is logged and turned into charts and a summary.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio web UI: drag-drop video in, annotated video + charts out |
| `engine.py` | Detection, tracking, counting, density, and chart generation |
| `requirements.txt` | Dependencies |

---

## License

MIT

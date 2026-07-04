import os
import json
from dataclasses import dataclass, field
from collections import defaultdict, deque

import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO


COCO_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
CLASS_COLORS = {
    "person": (0, 220, 255), "bicycle": (255, 180, 0), "car": (0, 230, 100),
    "motorcycle": (255, 120, 200), "bus": (0, 120, 255), "truck": (60, 60, 255),
}
CHART_CLASSES = ["car", "bus", "truck", "motorcycle", "person"]


@dataclass
class Config:
    model_path: str = "yolov8n.pt"
    confidence: float = 0.30
    iou: float = 0.50
    device: object = None
    img_size: int = 640
    target_class_ids: list = field(default_factory=lambda: list(COCO_CLASSES.keys()))
    line_start_frac: tuple = (0.0, 0.55)
    line_end_frac: tuple = (1.0, 0.55)
    density_low_max: int = 6
    density_high_min: int = 14
    density_smooth_window: int = 15
    output_dir: str = "outputs"
    max_width: int = 1280
    save_video: bool = True
    save_charts: bool = True
    save_csv: bool = True
    trace_length: int = 32
    draw_traces: bool = True


@dataclass
class Detection:
    track_id: object
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class YOLODetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)

    def detect(self, frame):
        results = self.model.track(
            frame, persist=True, conf=self.cfg.confidence, iou=self.cfg.iou,
            classes=self.cfg.target_class_ids, imgsz=self.cfg.img_size,
            device=self.cfg.device, tracker="bytetrack.yaml", verbose=False)
        r = results[0]
        out = []
        if r.boxes is None:
            return out
        ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else None
        for i, b in enumerate(r.boxes):
            cid = int(b.cls[0])
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            out.append(Detection(ids[i] if ids is not None else None, cid,
                COCO_CLASSES.get(cid, str(cid)), float(b.conf[0]), (x1, y1, x2, y2)))
        return out


class LineCounter:
    def __init__(self, line_start, line_end):
        self.line_start = line_start
        self.line_end = line_end
        self._last_side = {}
        self._counted = set()
        self.counts = defaultdict(lambda: {"in": 0, "out": 0})

    def _side(self, p):
        (x1, y1), (x2, y2) = self.line_start, self.line_end
        c = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
        return 1 if c > 0 else -1 if c < 0 else 0

    def _t(self, p):
        (x1, y1), (x2, y2) = self.line_start, self.line_end
        dx, dy = x2 - x1, y2 - y1
        d = dx * dx + dy * dy
        if d == 0:
            return -1.0
        return ((p[0] - x1) * dx + (p[1] - y1) * dy) / d

    def update(self, detections):
        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            side = self._side(det.center)
            if side == 0:
                continue
            prev = self._last_side.get(tid)
            self._last_side[tid] = side
            if prev is None or side == prev:
                continue
            if not (0.0 <= self._t(det.center) <= 1.0):
                continue
            if tid in self._counted:
                continue
            self._counted.add(tid)
            direction = "in" if side > 0 else "out"
            self.counts[det.class_name][direction] += 1

    def total_for(self, c):
        x = self.counts.get(c, {"in": 0, "out": 0})
        return x["in"] + x["out"]

    def grand_total(self):
        return sum(self.total_for(c) for c in self.counts)

    def as_dict(self):
        return {c: self.total_for(c) for c in self.counts}


class DensityMonitor:
    LEVEL_COLORS = {"Low": (0, 200, 0), "Medium": (0, 200, 255), "High": (0, 0, 255)}

    def __init__(self, cfg):
        self.low_max = cfg.density_low_max
        self.high_min = cfg.density_high_min
        self._window = deque(maxlen=max(1, cfg.density_smooth_window))

    def update(self, detections, frame_area):
        vehicles = [d for d in detections if d.class_name in VEHICLE_CLASSES]
        count = len(vehicles)
        self._window.append(count)
        smoothed = sum(self._window) / len(self._window)
        occupied = sum(d.area for d in vehicles)
        occupancy = min(occupied / frame_area if frame_area else 0.0, 1.0)
        level = self._level(smoothed)
        return {"vehicle_count": count, "smoothed_count": round(smoothed, 2),
                "occupancy": round(occupancy, 4), "level": level,
                "level_color": self.LEVEL_COLORS[level]}

    def _level(self, s):
        if s <= self.low_max:
            return "Low"
        if s >= self.high_min:
            return "High"
        return "Medium"


FONT = cv2.FONT_HERSHEY_SIMPLEX


class Annotator:
    def __init__(self, cfg):
        self.cfg = cfg
        self._traces = defaultdict(lambda: deque(maxlen=cfg.trace_length))

    def draw(self, frame, detections, line_start, line_end, counter, density):
        self._line(frame, line_start, line_end, density["level_color"])
        if self.cfg.draw_traces:
            self._trace(frame, detections)
        self._boxes(frame, detections)
        self._panel(frame, counter, density)
        return frame

    def _boxes(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_name, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = det.class_name + (f" #{det.track_id}" if det.track_id is not None else "") + f" {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 3, y1 - 5), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _trace(self, frame, detections):
        active = set()
        for det in detections:
            if det.track_id is None:
                continue
            active.add(det.track_id)
            self._traces[det.track_id].append(det.center)
            pts = list(self._traces[det.track_id])
            color = CLASS_COLORS.get(det.class_name, (200, 200, 200))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)
        for tid in list(self._traces.keys()):
            if tid not in active:
                del self._traces[tid]

    def _line(self, frame, start, end, color):
        cv2.line(frame, start, end, color, 3)
        cv2.putText(frame, "COUNT LINE", (start[0] + 8, max(start[1] - 8, 20)), FONT, 0.6, color, 2, cv2.LINE_AA)

    def _panel(self, frame, counter, density):
        x0, y0, pw, ph = 12, 12, 270, 200
        ov = frame.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + pw, y0 + ph), (25, 25, 25), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "TRAFFIC ANALYZER", (x0 + 12, y0 + 26), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x0 + 12, y0 + 36), (x0 + pw - 12, y0 + 36), (90, 90, 90), 1)
        y = y0 + 62
        for cls in ("car", "bus", "truck", "motorcycle", "person"):
            color = CLASS_COLORS.get(cls, (200, 200, 200))
            cv2.circle(frame, (x0 + 20, y - 4), 5, color, -1)
            cv2.putText(frame, f"{cls:<11}{counter.total_for(cls):>4}", (x0 + 34, y), FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
            y += 24
        cv2.putText(frame, "Density:", (x0 + 12, y + 4), FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, density["level"], (x0 + 110, y + 4), FONT, 0.6, density["level_color"], 2, cv2.LINE_AA)
        cv2.putText(frame, f"vehicles in frame: {density['vehicle_count']}", (x0 + 12, y + 26), FONT, 0.45, (190, 190, 190), 1, cv2.LINE_AA)


plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117", "savefig.facecolor": "#0e1117",
    "text.color": "#e6e6e6", "axes.labelcolor": "#e6e6e6", "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0", "axes.edgecolor": "#444444", "axes.grid": True,
    "grid.color": "#2a2f3a", "grid.linewidth": 0.6, "font.size": 11})


def _rgb(bgr):
    b, g, r = bgr
    return (r / 255, g / 255, b / 255)


class Analytics:
    def __init__(self, cfg, fps):
        self.cfg = cfg
        self.fps = fps if fps and fps > 0 else 30.0
        self.records = []

    def log_frame(self, idx, counter, density):
        rec = {"frame": idx, "time_s": round(idx / self.fps, 2),
               "vehicles_in_frame": density["vehicle_count"],
               "occupancy": density["occupancy"], "density_level": density["level"]}
        for c in CHART_CLASSES:
            rec[f"cum_{c}"] = counter.total_for(c)
        self.records.append(rec)

    def finalize(self, counter):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        df = pd.DataFrame(self.records)
        charts = []
        if self.cfg.save_csv and not df.empty:
            df.to_csv(os.path.join(self.cfg.output_dir, "traffic_log.csv"), index=False)
        if self.cfg.save_charts and not df.empty:
            charts = [self._totals(counter), self._share(counter),
                      self._cumulative(df), self._density(df)]
            charts = [c for c in charts if c]
        summary = self._summary(df, counter)
        with open(os.path.join(self.cfg.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary, charts

    def _save(self, fig, name):
        p = os.path.join(self.cfg.output_dir, name)
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        return p

    def _cumulative(self, df):
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in CHART_CLASSES:
            col = f"cum_{c}"
            if col in df and df[col].max() > 0:
                ax.plot(df["time_s"], df[col], label=c, linewidth=2, color=_rgb(CLASS_COLORS[c]))
        ax.set_title("Cumulative crossings over time", fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Count")
        ax.legend(facecolor="#161b22", edgecolor="#444")
        return self._save(fig, "chart_cumulative.png")

    def _totals(self, counter):
        data = [(c, counter.total_for(c)) for c in CHART_CLASSES if counter.total_for(c) > 0]
        if not data:
            return None
        labels, values = zip(*data)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=[_rgb(CLASS_COLORS[c]) for c in labels])
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontweight="bold")
        ax.set_title("Total crossings by type", fontweight="bold")
        ax.set_ylabel("Count")
        return self._save(fig, "chart_totals.png")

    def _share(self, counter):
        data = [(c, counter.total_for(c)) for c in CHART_CLASSES if counter.total_for(c) > 0]
        if not data:
            return None
        labels, values = zip(*data)
        fig, ax = plt.subplots(figsize=(7, 6))
        w, _, at = ax.pie(values, labels=labels, colors=[_rgb(CLASS_COLORS[c]) for c in labels],
            autopct="%1.0f%%", startangle=90, pctdistance=0.8,
            wedgeprops=dict(width=0.42, edgecolor="#0e1117"))
        for t in at:
            t.set_color("#0e1117")
            t.set_fontweight("bold")
        ax.set_title("Traffic composition", fontweight="bold")
        return self._save(fig, "chart_share.png")

    def _density(self, df):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df["time_s"], df["vehicles_in_frame"], color="#4da6ff", linewidth=1.6, label="vehicles in frame")
        low, high = self.cfg.density_low_max, self.cfg.density_high_min
        ax.axhspan(0, low, color="#00c853", alpha=0.10)
        ax.axhspan(low, high, color="#ffb300", alpha=0.10)
        ymax = max(df["vehicles_in_frame"].max(), high) + 2
        ax.axhspan(high, ymax, color="#ff5252", alpha=0.10)
        ax.axhline(low, color="#00c853", linewidth=0.8, linestyle="--")
        ax.axhline(high, color="#ff5252", linewidth=0.8, linestyle="--")
        ax.set_ylim(0, ymax)
        ax.set_title("Traffic density over time", fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Vehicles in frame")
        ax.legend(facecolor="#161b22", edgecolor="#444", loc="upper right")
        return self._save(fig, "chart_density.png")

    def _summary(self, df, counter):
        duration = float(df["time_s"].max()) if not df.empty else 0.0
        level_counts = df["density_level"].value_counts().to_dict() if not df.empty else {}
        busiest = ""
        if not df.empty:
            row = df.loc[df["vehicles_in_frame"].idxmax()]
            busiest = f"{row['vehicles_in_frame']} vehicles at {row['time_s']}s"
        return {"duration_seconds": round(duration, 1), "frames_processed": len(df),
                "total_crossings": counter.grand_total(), "crossings_by_class": counter.as_dict(),
                "directional": {c: dict(counter.counts[c]) for c in counter.counts},
                "busiest_moment": busiest, "density_distribution_frames": level_counts}


def frac_point(frac, w, h):
    return int(frac[0] * w), int(frac[1] * h)


def open_capture(input_path):
    source = 0 if str(input_path) == "0" else input_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {input_path}")
    meta = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "total": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0}
    return cap, meta


def analyze_video(cfg, input_path, progress=None, max_frames=None):
    cap, meta = open_capture(input_path)
    width, height, fps, total = meta["width"], meta["height"], meta["fps"], meta["total"]

    scale = 1.0
    if cfg.max_width and width > cfg.max_width:
        scale = cfg.max_width / width
        width, height = int(width * scale), int(height * scale)

    frame_area = width * height
    line_start = frac_point(cfg.line_start_frac, width, height)
    line_end = frac_point(cfg.line_end_frac, width, height)

    os.makedirs(cfg.output_dir, exist_ok=True)
    detector = YOLODetector(cfg)
    counter = LineCounter(line_start, line_end)
    density_monitor = DensityMonitor(cfg)
    annotator = Annotator(cfg)
    analytics = Analytics(cfg, fps)

    out_path = os.path.join(cfg.output_dir, "annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)) if cfg.save_video else None

    if max_frames:
        total = min(total, max_frames) if total else max_frames

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (width, height))
            dets = detector.detect(frame)
            counter.update(dets)
            density = density_monitor.update(dets, frame_area)
            annotator.draw(frame, dets, line_start, line_end, counter, density)
            analytics.log_frame(idx, counter, density)
            if writer is not None:
                writer.write(frame)
            idx += 1
            if progress is not None:
                progress(idx, total)
            if max_frames and idx >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    summary, charts = analytics.finalize(counter)
    summary["output_video"] = out_path if cfg.save_video else None
    summary["fps"] = fps
    summary["resolution"] = [width, height]
    return summary, charts

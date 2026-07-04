import os
import shutil
import subprocess
import uuid

import cv2
import gradio as gr

from engine import Config, analyze_video


MAX_SECONDS = 30
WORK_DIR = "runs"
os.makedirs(WORK_DIR, exist_ok=True)


def _to_h264(src_path, dst_path):
    if not shutil.which("ffmpeg"):
        return src_path
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst_path],
            check=True, capture_output=True)
        return dst_path
    except subprocess.CalledProcessError:
        return src_path


def analyze(video_path, model_choice, confidence, line_pos, max_seconds,
            progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload a video first.")

    run_dir = os.path.join(WORK_DIR, uuid.uuid4().hex[:8])
    os.makedirs(run_dir, exist_ok=True)

    cfg = Config()
    cfg.model_path = "yolov8s.pt" if model_choice == "Accurate (yolov8s)" else "yolov8n.pt"
    cfg.confidence = float(confidence)
    cfg.line_start_frac = (0.0, float(line_pos))
    cfg.line_end_frac = (1.0, float(line_pos))
    cfg.output_dir = run_dir

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    cap_seconds = min(int(max_seconds), MAX_SECONDS)
    max_frames = int(fps * cap_seconds)

    progress(0, desc="Loading model…")

    def on_progress(i, total):
        if total:
            progress(min(i / total, 1.0), desc=f"Analyzing frame {i}/{total}")
        else:
            progress(0.5, desc=f"Analyzing frame {i}")

    summary, charts = analyze_video(cfg, video_path, progress=on_progress,
                                    max_frames=max_frames)

    web_video = _to_h264(summary["output_video"],
                         os.path.join(run_dir, "annotated_web.mp4"))

    counts = summary["crossings_by_class"]
    lines = [f"**Total crossings: {summary['total_crossings']}**", ""]
    for cls in ["car", "bus", "truck", "motorcycle", "person"]:
        if cls in counts:
            lines.append(f"- {cls}: {counts[cls]}")
    lines.append("")
    lines.append(f"Duration analyzed: {summary['duration_seconds']}s "
                 f"({summary['frames_processed']} frames)")
    if summary["busiest_moment"]:
        lines.append(f"Busiest moment: {summary['busiest_moment']}")
    summary_md = "\n".join(lines)

    return web_video, charts, summary_md


with gr.Blocks(title="Road Traffic Analyzer") as demo:
    gr.Markdown(
        "# 🚦 Road Traffic Analyzer\n"
        "Drop in a street video. Get back an annotated video that detects and "
        "counts cars, buses, trucks, motorcycles, and people, with live traffic "
        "density — plus analytics charts. Powered by YOLOv8."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Drop your video here", sources=["upload"])
            model_choice = gr.Radio(
                ["Fast (yolov8n)", "Accurate (yolov8s)"],
                value="Fast (yolov8n)", label="Model")
            confidence = gr.Slider(0.1, 0.7, value=0.30, step=0.05,
                                   label="Detection confidence")
            line_pos = gr.Slider(0.2, 0.8, value=0.55, step=0.05,
                                 label="Counting line position (top→bottom)")
            max_seconds = gr.Slider(5, MAX_SECONDS, value=15, step=5,
                                    label="Seconds to analyze")
            run_btn = gr.Button("Analyze traffic", variant="primary")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Annotated video", autoplay=True)
            summary_out = gr.Markdown()
            gallery_out = gr.Gallery(label="Analytics", columns=2, height="auto")

    run_btn.click(
        analyze,
        inputs=[video_in, model_choice, confidence, line_pos, max_seconds],
        outputs=[video_out, gallery_out, summary_out],
    )

    gr.Markdown(
        "Tip: use a clip where the camera looks down the road, and place the "
        "counting line where vehicles clearly cross it. Shorter clips finish faster."
    )


if __name__ == "__main__":
    try:
        demo.queue(max_size=10).launch(
            server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
    except TypeError:
        demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860)

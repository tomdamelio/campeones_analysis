"""
Extract luminance values from green intensity videos.

For each .mp4 video in stimuli/luminance/, this script reads every frame,
extracts the mean green channel value (the only non-zero channel in these
videos), and saves a CSV with columns [timestamp, luminance] to the same
directory.

Usage
-----
    python scripts/extract_luminance_from_videos.py
"""

import csv
import sys
from pathlib import Path

import cv2  # opencv-python


def extract_luminance(video_path: Path, output_csv: Path) -> None:
    """Read *video_path* frame-by-frame and write green-channel luminance.

    Parameters
    ----------
    video_path : Path
        Path to the input .mp4 video.
    output_csv : Path
        Path where the resulting .csv will be saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps:.2f}  |  Total frames: {total_frames}")

    rows: list[tuple[float, float]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Green channel is index 1 in OpenCV's BGR order.
        # These videos only have green values (R=0, B=0).
        mean_green = frame[:, :, 1].mean()

        timestamp = frame_idx / fps
        rows.append((timestamp, mean_green))
        frame_idx += 1

    cap.release()

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "luminance"])
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows -> {output_csv}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    luminance_dir = project_root / "stimuli" / "luminance"

    if not luminance_dir.is_dir():
        sys.exit(f"Directory not found: {luminance_dir}")

    video_files = sorted(luminance_dir.glob("green_intensity_video_*.mp4"))
    if not video_files:
        sys.exit(f"No green_intensity_video_*.mp4 files found in {luminance_dir}")

    print(f"Found {len(video_files)} video(s) in {luminance_dir}\n")

    for video_path in video_files:
        csv_name = video_path.stem + ".csv"
        output_csv = luminance_dir / csv_name
        print(f"Processing: {video_path.name}")
        extract_luminance(video_path, output_csv)
        print()

    print("Done.")


if __name__ == "__main__":
    main()

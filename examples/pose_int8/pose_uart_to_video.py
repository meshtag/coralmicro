#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import re
import shutil
import subprocess
import sys

try:
    import serial  # type: ignore
except ImportError:
    serial = None

KEYPOINT_NAMES = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
]
KEYPOINT_NAME_TO_ID = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}
BODY_PARTS = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (2, 16),
    (5, 17),
]
LINE_COLOR = (255, 255, 0)
POINT_COLOR = (255, 32, 32)
POINT_OUTLINE_COLOR = (255, 255, 255)
BBOX_COLOR = (0, 255, 0)

INPUT_RE = re.compile(r"^Input:\s+(\d+)x(\d+)x(\d+)\s")
POSES_RE = re.compile(r"^Poses\s+\((\d+)\)")
POSES_NONE_RE = re.compile(r"^Poses:\s+none")
POSE_RE = re.compile(r"^Pose\s+\d+\s+conf=[0-9.]+\s+kpts=\d+")
KPT_RE = re.compile(r"^([A-Za-z]+)\s+x=\s*(-?\d+)\s+y=\s*(-?\d+)")


def open_ffmpeg(output_path, width, height, fps):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it and retry.")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def open_ffmpeg_overlay(output_path, background_path, skel_w, skel_h, fps,
                        bg_w, bg_h):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found. Install it and retry.")
    filter_graph = (
        f"[1:v]scale={bg_w}:{bg_h},colorkey=0x000000:0.1:0.0[sk];"
        "[0:v][sk]overlay=shortest=1"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        background_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{skel_w}x{skel_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-filter_complex",
        filter_graph,
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def probe_video_size(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    width_str, height_str = out.split("x")
    return int(width_str), int(height_str)


def set_pixel(buf, width, height, x, y, color):
    if 0 <= x < width and 0 <= y < height:
        idx = (y * width + x) * 3
        buf[idx] = color[0]
        buf[idx + 1] = color[1]
        buf[idx + 2] = color[2]


def draw_circle(buf, width, height, cx, cy, radius, color):
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r2:
                set_pixel(buf, width, height, cx + dx, cy + dy, color)


def draw_line(buf, width, height, x0, y0, x1, y1, color, thickness):
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        draw_circle(buf, width, height, x0, y0, thickness, color)
        return
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + dx * t))
        y = int(round(y0 + dy * t))
        draw_circle(buf, width, height, x, y, thickness, color)


def draw_bbox(buf, width, height, points, color, thickness):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    draw_line(buf, width, height, min_x, min_y, max_x, min_y, color, thickness)
    draw_line(buf, width, height, max_x, min_y, max_x, max_y, color, thickness)
    draw_line(buf, width, height, max_x, max_y, min_x, max_y, color, thickness)
    draw_line(buf, width, height, min_x, max_y, min_x, min_y, color, thickness)


def render_frame(width, height, poses, scale_x, scale_y, out_w, out_h,
                 draw_boxes):
    buf = bytearray(out_w * out_h * 3)
    scale = (scale_x + scale_y) / 2.0
    point_radius = max(3, int(round(scale * 1.1)))
    line_thickness = max(2, int(round(scale * 0.9)))
    point_outline = max(1, int(round(point_radius * 0.35)))
    for pose in poses:
        kpts = pose["kpts"]
        for a, b in BODY_PARTS:
            if a in kpts and b in kpts:
                x0, y0 = kpts[a]
                x1, y1 = kpts[b]
                sx0 = int(round(x0 * scale_x))
                sy0 = int(round(y0 * scale_y))
                sx1 = int(round(x1 * scale_x))
                sy1 = int(round(y1 * scale_y))
                draw_line(buf, out_w, out_h, sx0, sy0, sx1, sy1, LINE_COLOR,
                          line_thickness)
        points = []
        for _, (x, y) in kpts.items():
            sx = int(round(x * scale_x))
            sy = int(round(y * scale_y))
            points.append((sx, sy))
            if point_outline > 0:
                draw_circle(buf, out_w, out_h, sx, sy,
                            point_radius + point_outline,
                            POINT_OUTLINE_COLOR)
            draw_circle(buf, out_w, out_h, sx, sy, point_radius, POINT_COLOR)
        if draw_boxes and len(points) >= 2:
            draw_bbox(buf, out_w, out_h, points, BBOX_COLOR, line_thickness)
    return buf, out_w, out_h


def iter_lines(args):
    if args.input:
        with open(args.input, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
    elif args.serial:
        if serial is None:
            raise RuntimeError("pyserial not installed. Install it or use --input.")
        with serial.Serial(args.serial, args.baud, timeout=1) as ser:
            while True:
                line = ser.readline()
                if not line:
                    continue
                yield line.decode("utf-8", errors="replace")
    else:
        for line in sys.stdin:
            yield line


def parse_frames(lines, width, height):
    current_poses = []
    current_pose = None
    in_frame = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = INPUT_RE.match(line)
        if m:
            height = int(m.group(1))
            width = int(m.group(2))
            continue
        if POSES_NONE_RE.match(line) or POSES_RE.match(line):
            if in_frame:
                yield current_poses, width, height
            in_frame = True
            current_poses = []
            current_pose = None
            continue
        if POSE_RE.match(line):
            current_pose = {"kpts": {}}
            current_poses.append(current_pose)
            continue
        m = KPT_RE.match(line)
        if m and current_pose is not None:
            name = m.group(1)
            if name in KEYPOINT_NAME_TO_ID:
                idx = KEYPOINT_NAME_TO_ID[name]
                x = int(m.group(2))
                y = int(m.group(3))
                current_pose["kpts"][idx] = (x, y)
    if in_frame:
        yield current_poses, width, height


def parse_frames_jsonl(path, width, height):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            frame_width = data.get("width", width)
            frame_height = data.get("height", height)
            if frame_width is None or frame_height is None:
                raise SystemExit("Missing width/height in JSONL data.")
            poses = []
            for pose in data.get("poses", []):
                kpts = {}
                keypoints = pose.get("keypoints", [])
                for idx, pair in enumerate(keypoints):
                    if not isinstance(pair, list) or len(pair) != 2:
                        continue
                    x, y = pair
                    if x < 0 or y < 0:
                        continue
                    kpts[idx] = (int(x), int(y))
                poses.append({"kpts": kpts})
            yield poses, frame_width, frame_height


def main():
    parser = argparse.ArgumentParser(
        description="Convert pose_int8 UART output into a skeleton video."
    )
    parser.add_argument("--input", help="UART log file (text).")
    parser.add_argument("--input-jsonl", help="Pose JSONL file.")
    parser.add_argument("--serial", help="Serial port (e.g. /dev/cu.usbmodem123).")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--output", required=True, help="Output video path (mp4).")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--background", help="Background video to overlay.")
    parser.add_argument("--width", type=int, help="Override model input width.")
    parser.add_argument("--height", type=int, help="Override model input height.")
    parser.add_argument("--max-frames", type=int, help="Stop after N frames.")
    parser.add_argument("--no-bbox", action="store_true", help="Disable bounding box.")
    args = parser.parse_args()

    if args.scale < 1:
        raise SystemExit("--scale must be >= 1")
    if args.input and args.serial:
        raise SystemExit("Use only one of --input or --serial.")
    if args.input_jsonl and (args.input or args.serial):
        raise SystemExit("Use --input-jsonl without --input/--serial.")

    width = args.width
    height = args.height
    writer = None
    frame_count = 0
    bg_w = None
    bg_h = None
    last_poses = None
    if args.background:
        bg_w, bg_h = probe_video_size(args.background)

    try:
        if args.input_jsonl:
            frame_iter = parse_frames_jsonl(args.input_jsonl, width, height)
        else:
            frame_iter = parse_frames(iter_lines(args), width, height)
        for poses, width, height in frame_iter:
            if poses:
                last_poses = poses
            elif last_poses is not None:
                poses = last_poses
            if width is None or height is None:
                raise SystemExit(
                    "Missing model size. Provide --width/--height or include the Input line."
                )
            if args.background:
                scale_x = bg_w / float(width)
                scale_y = bg_h / float(height)
                out_w = bg_w
                out_h = bg_h
            else:
                scale_x = args.scale
                scale_y = args.scale
                out_w = width * args.scale
                out_h = height * args.scale
            frame, out_w, out_h = render_frame(
                width, height, poses, scale_x, scale_y, out_w, out_h,
                draw_boxes=not args.no_bbox,
            )
            if writer is None:
                if args.background:
                    writer = open_ffmpeg_overlay(
                        args.output,
                        args.background,
                        out_w,
                        out_h,
                        args.fps,
                        bg_w,
                        bg_h,
                    )
                else:
                    writer = open_ffmpeg(args.output, out_w, out_h, args.fps)
            if writer.stdin is None:
                raise SystemExit("ffmpeg stdin unavailable.")
            writer.stdin.write(frame)
            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer and writer.stdin:
            writer.stdin.close()
            writer.wait()

    if frame_count == 0:
        raise SystemExit("No frames parsed. Check UART output or log file.")


if __name__ == "__main__":
    main()

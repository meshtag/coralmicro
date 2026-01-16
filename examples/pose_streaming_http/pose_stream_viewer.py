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
import io
import json
import time
import urllib.request

try:
    import tkinter as tk
except ImportError as exc:
    tk = None
    TK_IMPORT_ERROR = exc
else:
    TK_IMPORT_ERROR = None

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError as exc:
    Image = None
    ImageDraw = None
    ImageTk = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None

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


def fetch_bytes(url, timeout, retries):
    last_error = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    raise SystemExit(f"Failed to fetch {url}: {last_error}")


def parse_pose_data(pose_data):
    poses = []
    for pose in pose_data.get("poses", []):
        kpts = {}
        keypoints = pose.get("keypoints", [])
        for idx, pair in enumerate(keypoints):
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            x, y = pair
            if x < 0 or y < 0:
                continue
            kpts[idx] = (int(x), int(y))
        poses.append(kpts)
    return poses, pose_data.get("width"), pose_data.get("height")


def draw_poses(image, poses, scale_x, scale_y):
    draw = ImageDraw.Draw(image)
    scale = (scale_x + scale_y) / 2.0
    point_radius = max(3, int(round(scale * 1.1)))
    line_width = max(2, int(round(scale * 0.9)))
    outline = max(1, int(round(point_radius * 0.35)))

    for kpts in poses:
        for a, b in BODY_PARTS:
            if a in kpts and b in kpts:
                x0, y0 = kpts[a]
                x1, y1 = kpts[b]
                draw.line(
                    (
                        x0 * scale_x,
                        y0 * scale_y,
                        x1 * scale_x,
                        y1 * scale_y,
                    ),
                    fill=LINE_COLOR,
                    width=line_width,
                )
        for x, y in kpts.values():
            sx = x * scale_x
            sy = y * scale_y
            if outline > 0:
                draw.ellipse(
                    (
                        sx - (point_radius + outline),
                        sy - (point_radius + outline),
                        sx + (point_radius + outline),
                        sy + (point_radius + outline),
                    ),
                    fill=POINT_OUTLINE_COLOR,
                )
            draw.ellipse(
                (
                    sx - point_radius,
                    sy - point_radius,
                    sx + point_radius,
                    sy + point_radius,
                ),
                fill=POINT_COLOR,
            )


def ensure_deps():
    if Image is None or ImageDraw is None or ImageTk is None:
        raise SystemExit(
            "Pillow is required. Run: python3 -m pip install -r "
            "examples/pose_streaming_http/requirements.txt"
        )
    if tk is None:
        raise SystemExit(f"tkinter is required: {TK_IMPORT_ERROR}")


def main():
    parser = argparse.ArgumentParser(
        description="View pose_streaming_http camera frames with pose overlay."
    )
    parser.add_argument("--base-url", required=True, help="e.g. http://10.10.10.1")
    parser.add_argument("--pose-endpoint", default="/pose_data")
    parser.add_argument("--raw-endpoint", default="/camera_stream_raw")
    parser.add_argument("--overlay-endpoint", default="/camera_stream")
    parser.add_argument(
        "--use-board-overlay",
        action="store_true",
        help="Use the board-drawn overlay (skip host drawing).",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--hold-last", dest="hold_last", action="store_true")
    parser.add_argument("--no-hold-last", dest="hold_last", action="store_false")
    parser.set_defaults(hold_last=True)
    args = parser.parse_args()

    ensure_deps()

    if args.scale <= 0:
        raise SystemExit("--scale must be > 0")

    base_url = args.base_url.rstrip("/")
    if args.use_board_overlay:
        frame_url = f"{base_url}{args.overlay_endpoint}"
    else:
        frame_url = f"{base_url}{args.raw_endpoint}"
        pose_url = f"{base_url}{args.pose_endpoint}"

    root = tk.Tk()
    root.title("Dev Board Micro Pose Stream")
    label = tk.Label(root)
    label.pack()

    should_stop = [False]

    def on_close():
        should_stop[0] = True
        root.destroy()

    def on_key(event):
        if event.char.lower() == "q":
            on_close()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Key>", on_key)

    last_poses = None
    last_pose_size = None
    frame_count = 0

    try:
        while not should_stop[0]:
            start = time.monotonic()
            jpeg = fetch_bytes(frame_url, args.timeout, args.retries)
            image = Image.open(io.BytesIO(jpeg)).convert("RGB")
            orig_w, orig_h = image.size

            if args.scale != 1.0:
                image = image.resize(
                    (int(orig_w * args.scale), int(orig_h * args.scale)),
                    Image.BILINEAR,
                )

            if not args.use_board_overlay:
                poses = []
                pose_w = orig_w
                pose_h = orig_h
                pose_bytes = fetch_bytes(pose_url, args.timeout, args.retries)
                pose_text = pose_bytes.decode("utf-8", errors="replace").strip()
                if pose_text:
                    pose_data = json.loads(pose_text)
                    poses, pose_w, pose_h = parse_pose_data(pose_data)
                    if pose_w is None or pose_h is None:
                        pose_w, pose_h = orig_w, orig_h
                    if poses:
                        last_poses = poses
                        last_pose_size = (pose_w, pose_h)
                    elif args.hold_last and last_poses is not None:
                        poses = last_poses
                        if last_pose_size:
                            pose_w, pose_h = last_pose_size
                elif args.hold_last and last_poses is not None:
                    poses = last_poses
                    if last_pose_size:
                        pose_w, pose_h = last_pose_size

                if poses:
                    scale_x = image.width / float(pose_w)
                    scale_y = image.height / float(pose_h)
                    draw_poses(image, poses, scale_x, scale_y)

            tk_image = ImageTk.PhotoImage(image=image)
            label.configure(image=tk_image)
            label.image = tk_image

            root.update_idletasks()
            root.update()

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break

            if args.fps > 0:
                elapsed = time.monotonic() - start
                delay = max(0.0, (1.0 / args.fps) - elapsed)
                if delay:
                    time.sleep(delay)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

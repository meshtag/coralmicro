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
import shutil
import subprocess
import time
import urllib.request


def open_ffmpeg(output_path, fps):
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. Install it and retry.")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "mjpeg",
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


def main():
    parser = argparse.ArgumentParser(
        description="Capture pose_streaming_http video + pose JSONL on host."
    )
    parser.add_argument("--base-url", required=True, help="e.g. http://10.10.10.1")
    parser.add_argument("--out-video", required=True, help="Output mp4 path.")
    parser.add_argument("--out-poses", required=True, help="Output JSONL path.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--raw-endpoint", default="/camera_stream_raw")
    parser.add_argument("--pose-endpoint", default="/pose_data")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    raw_url = f"{base_url}{args.raw_endpoint}"
    pose_url = f"{base_url}{args.pose_endpoint}"

    writer = open_ffmpeg(args.out_video, args.fps)
    if writer.stdin is None:
        raise SystemExit("ffmpeg stdin unavailable.")

    frame_idx = 0
    try:
        with open(args.out_poses, "w", encoding="utf-8") as pose_file:
            while True:
                if args.max_frames and frame_idx >= args.max_frames:
                    break
                jpeg = fetch_bytes(raw_url, args.timeout, args.retries)
                pose_bytes = fetch_bytes(pose_url, args.timeout, args.retries)
                pose_text = pose_bytes.decode("utf-8", errors="replace").strip()
                if not pose_text:
                    continue
                pose_data = json.loads(pose_text)
                pose_data["host_frame"] = frame_idx
                pose_data["host_time_ms"] = int(time.time() * 1000)
                pose_file.write(json.dumps(pose_data) + "\n")
                writer.stdin.write(jpeg)
                frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        if writer.stdin:
            writer.stdin.close()
            writer.wait()


if __name__ == "__main__":
    main()

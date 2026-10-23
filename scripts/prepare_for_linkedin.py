#!/usr/bin/env python3
"""
Convert an MP4 video to LinkedIn-compatible format:
1. Adds a silent AAC stereo audio track (LinkedIn processing often hangs/fails without audio).
2. Enables +faststart (places moov atom at beginning for web streaming and chunked upload).
3. Preserves high quality H.264 video.
"""

import sys
import subprocess
from pathlib import Path
import imageio_ffmpeg


def prepare_for_linkedin(input_path: str, output_path: str | None = None) -> Path:
    in_file = Path(input_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    if output_path is None:
        out_file = in_file.with_name(f"{in_file.stem}_linkedin{in_file.suffix}")
    else:
        out_file = Path(output_path)

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(in_file),
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(out_file),
    ]

    print(f"Converting '{in_file}' -> '{out_file}'...")
    subprocess.run(cmd, check=True)
    print(f"Successfully generated LinkedIn-compatible video: {out_file}")
    return out_file


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/hoverslam_demo.mp4"
    dst = sys.argv[2] if len(sys.argv) > 2 else None
    prepare_for_linkedin(src, dst)

"""POST every file in ./audio to the local transcribe API and print output paths."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

ALLOWED = {".mp3", ".m4a", ".flac", ".wav"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Transcribe API base URL",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "audio",
        help="Folder containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "shared" / "output",
        help="Where .md results are written (must match server SHARED_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between task status polls",
    )
    args = parser.parse_args()

    root: Path = args.audio_dir
    if not root.is_dir():
        print(f"Audio folder not found: {root}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED
    )
    if not paths:
        print(f"No audio in {root} (use: {', '.join(sorted(ALLOWED))})")
        return

    base = args.base_url.rstrip("/")
    out_dir: Path = args.output_dir

    with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        for path in paths:
            print(f"\n=== {path.name} ===")
            with path.open("rb") as f:
                r = client.post(
                    f"{base}/transcribe",
                    files={"file": (path.name, f, "application/octet-stream")},
                )
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                print(e.response.text, file=sys.stderr)
                raise
            body = r.json()
            task_id = body["task_id"]
            print("task_id:", task_id)

            deadline = time.monotonic() + 3600.0
            while time.monotonic() < deadline:
                s = client.get(f"{base}/tasks/{task_id}")
                s.raise_for_status()
                st = s.json()
                if st.get("ready"):
                    mp = out_dir / f"{task_id}.md"
                    print("result:", mp)
                    if mp.is_file():
                        text = mp.read_text(encoding="utf-8")
                        if "---\n\n" in text:
                            body = text.split("---\n\n", 1)[-1]
                        else:
                            body = text
                        preview = body[:500]
                        if len(body) > 500:
                            preview += "…"
                        print("transcript preview:\n", preview)
                    break
                if st.get("audio_saved") and not st.get("ready"):
                    af = st.get("audio_file")
                    print(
                        "transcript failed or pending, source audio kept:",
                        out_dir / af if af else "?",
                        file=sys.stderr,
                    )
                    break
                time.sleep(args.poll_interval)
            else:
                print("Timeout waiting for task", task_id, file=sys.stderr)


if __name__ == "__main__":
    main()

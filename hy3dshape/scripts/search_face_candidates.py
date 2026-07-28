"""Run a small, reproducible seed search for face-safe 2mv geometry."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from gradio_client import Client, handle_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--octree", type=int, default=256)
    parser.add_argument("--chunks", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = Client(args.server, verbose=False)
    inputs = {
        view: args.input_dir / f"Gohan_{view.capitalize()}_transparent.png"
        for view in ("front", "back", "left", "right")
    }
    for view, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{view} view not found: {path}")

    records: list[dict[str, object]] = []
    for index, seed in enumerate(args.seeds, start=1):
        started = time.perf_counter()
        print(
            f"[{index}/{len(args.seeds)}] seed={seed} steps={args.steps} "
            f"guidance={args.guidance} octree={args.octree}",
            flush=True,
        )
        generated_mesh, viewer_html, stats, actual_seed = client.predict(
            "four",
            None,
            handle_file(inputs["front"]),
            handle_file(inputs["back"]),
            handle_file(inputs["left"]),
            handle_file(inputs["right"]),
            args.steps,
            args.guidance,
            seed,
            args.octree,
            False,
            args.chunks,
            False,
            api_name="/shape_generation",
        )
        html_match = re.search(r"generation-viewer/([0-9a-f-]{36})", viewer_html)
        generation_uid = html_match.group(1) if html_match else None
        if generation_uid is None and isinstance(stats, dict):
            generation_uid = stats.get("generation", {}).get("uid")
        record = {
            "seed": int(actual_seed),
            "steps": args.steps,
            "guidance": args.guidance,
            "octree": args.octree,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "generation_uid": generation_uid,
            "mesh": str(generated_mesh),
            "stats": stats,
        }
        records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(
            f"  completed uid={generation_uid} in {record['elapsed_seconds']}s",
            flush=True,
        )

    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

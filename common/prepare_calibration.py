"""Create one shared 640x640 letterboxed calibration set for both toolchains.

No files are written unless --run-conversion is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[Path]:
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            items.append(Path(value).expanduser().resolve())
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rknn-list", type=Path, required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--run-conversion", action="store_true")
    args = parser.parse_args()

    sources = read_manifest(args.manifest)
    print(f"images={len(sources)}, size={args.size}, output={args.output_dir.resolve()}")
    if not args.run_conversion:
        print("DRY RUN: no calibration images were created")
        return
    if not sources:
        raise RuntimeError("calibration manifest is empty")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to write into non-empty directory: {args.output_dir}")

    import cv2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    generated = []
    for index, source in enumerate(sources):
        if not source.is_file():
            raise FileNotFoundError(source)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {source}")
        height, width = image.shape[:2]
        scale = min(args.size / width, args.size / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        left = (args.size - new_width) // 2
        top = (args.size - new_height) // 2
        output = cv2.copyMakeBorder(
            resized,
            top,
            args.size - new_height - top,
            left,
            args.size - new_width - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        target = output_dir / f"{index:05d}.png"
        if not cv2.imwrite(str(target), output):
            raise RuntimeError(f"failed to write image: {target}")
        generated.append(str(target))
        records.append(
            {
                "source": str(source),
                "source_sha256": sha256(source),
                "output": str(target),
                "output_sha256": sha256(target),
                "original_size": [width, height],
                "resized_size": [new_width, new_height],
                "padding_left_top": [left, top],
            }
        )

    args.rknn_list.parent.mkdir(parents=True, exist_ok=True)
    args.rknn_list.write_text("\n".join(generated) + "\n", encoding="utf-8")
    report = output_dir.parent / "calibration_manifest.generated.json"
    report.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated={len(generated)}")
    print(f"rknn_list={args.rknn_list.resolve()}")
    print(f"report={report.resolve()}")


if __name__ == "__main__":
    main()

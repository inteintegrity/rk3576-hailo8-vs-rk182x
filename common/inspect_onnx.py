"""Read-only structural inspection for the canonical ONNX."""

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


def tensor_info(value) -> dict:
    tensor_type = value.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param or "dynamic")
    return {"name": value.name, "elem_type": tensor_type.elem_type, "shape": dims}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    args = parser.parse_args()

    import onnx

    path = args.onnx.resolve()
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    report = {
        "path": str(path),
        "sha256": sha256(path),
        "opset": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "inputs": [tensor_info(item) for item in model.graph.input],
        "outputs": [tensor_info(item) for item in model.graph.output],
        "node_count": len(model.graph.node),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


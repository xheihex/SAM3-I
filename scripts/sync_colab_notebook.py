#!/usr/bin/env python3
"""Synchronize the notebook's standalone helpers with their reviewed source modules."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/SAM3I_single_image_colab.ipynb"


def embedded_source(name):
    module = (ROOT / name).read_text(encoding="utf-8")
    digest = hashlib.sha256(module.encode("utf-8")).hexdigest()
    title = "部署辅助函数（运行一次）" if "helpers" in name else "编号与置信度展示函数（运行一次）"
    return (
        f"# @title {title}\n"
        + module.rstrip() + "\n\n"
        + 'EMBEDDED_SOURCE_SHA256 = dict(globals().get("EMBEDDED_SOURCE_SHA256", {}))\n'
        + f"EMBEDDED_SOURCE_SHA256[{name!r}] = {digest!r}\n"
    )


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if name := cell.get("metadata", {}).get("sam3i_source"):
            cell["source"] = embedded_source(name).splitlines(keepends=True)
            cell["execution_count"] = None
            cell["outputs"] = []
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()

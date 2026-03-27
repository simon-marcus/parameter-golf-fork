#!/usr/bin/env python3
import importlib
import json
import os
import shutil
import sys


def version_or_none(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "version": getattr(module, "__version__", "unknown")}


report = {
    "python": sys.version.split()[0],
    "cwd": os.getcwd(),
    "bins": {name: shutil.which(name) for name in ["python3", "git", "rsync", "tmux", "jq", "pigz"]},
    "modules": {
        "torch": version_or_none("torch"),
        "numpy": version_or_none("numpy"),
        "sentencepiece": version_or_none("sentencepiece"),
        "zstandard": version_or_none("zstandard"),
        "huggingface_hub": version_or_none("huggingface_hub"),
    },
    "paths": {
        "workspace_exists": os.path.isdir("/workspace"),
        "tmp_exists": os.path.isdir("/tmp"),
        "report_script_exists": os.path.exists("/opt/parameter-golf/preflight_report.py"),
    },
}

if report["modules"]["torch"]["ok"]:
    import torch

    report["torch_cuda_available"] = torch.cuda.is_available()
    report["torch_version"] = torch.__version__
else:
    report["torch_cuda_available"] = False

print(json.dumps(report, indent=2, sort_keys=True))

missing = [name for name, meta in report["modules"].items() if not meta["ok"]]
if missing:
    raise SystemExit(f"missing python modules: {', '.join(missing)}")

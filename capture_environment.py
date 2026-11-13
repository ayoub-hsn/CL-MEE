#!/usr/bin/env python3
"""Capture interpreter, platform, and installed package details for auditing."""
from __future__ import annotations
import argparse, importlib.metadata, json, platform, subprocess, sys
from pathlib import Path

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "notebook", "ipykernel", "nbformat", "pypdf"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("environment_capture"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Python: {sys.version.replace(chr(10), ' ')}",
        f"Implementation: {platform.python_implementation()}",
        f"Platform: {platform.platform()}",
        f"System: {platform.system()} {platform.release()}",
        f"Machine: {platform.machine()}",
        "Installed package versions:",
    ]
    for name in PACKAGES:
        try:
            lines.append(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{name}: NOT INSTALLED")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            check=True, capture_output=True, text=True).stdout
    (args.output_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    package_versions = {}
    for name in PACKAGES:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    metadata = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": package_versions,
    }
    (args.output_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Environment captured in", args.output_dir.resolve())

if __name__ == "__main__":
    main()

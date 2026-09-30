"""Export Git-visible sources with an integrity manifest, excluding local outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_ROOTS = {"results", "third_party", "build", "dist"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT.parent / "PrivFim-git.zip")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("Write the release archive outside the source directory")
    if output.exists():
        parser.error(f"Archive already exists: {output}; choose a new path")
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    manifest = []
    selected = []
    for name in sorted(set(files) - {""}):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source path: {name}")
        if relative.parts[0] in EXCLUDED_ROOTS or any(
            part.startswith(".venv") or part in {".git", "__pycache__", ".pytest_cache"}
            for part in relative.parts
        ) or relative.parts[:2] in {("data", "raw"), ("data", "real")}:
            raise ValueError(f"Generated file unexpectedly visible to Git: {name}")
        source = ROOT / relative
        if source.is_symlink():
            raise ValueError(f"Resolve symlink before packaging: {name}")
        if not source.exists():
            continue
        if source.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f"Review large source before packaging: {name}")
        content = source.read_bytes()
        manifest.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        selected.append((name, content))
    if not any(name == "pyproject.toml" for name, _ in selected):
        raise ValueError("Missing project sources. Run git init in the repository first.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in selected:
            archive.writestr("PrivFim/" + name, content)
        archive.writestr("PrivFim/RELEASE_MANIFEST.json", json.dumps(
            {"format": 1, "files": manifest}, ensure_ascii=False, indent=2
        ) + "\n")
    print(json.dumps({"archive": str(output), "files": len(selected),
                      "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()

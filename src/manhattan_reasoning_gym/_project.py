"""Package only explicitly selected source files for a cloud build."""

import io
import json
import re
import zipfile
from pathlib import Path


def source_bundle(design: str, sources: list[str]) -> bytes:
    entry = Path(design).absolute()
    root = entry.parent.resolve()
    selected = [entry.name, *sources]
    if entry.suffix not in {".py", ".v"}:
        raise ValueError("The main design must be a .py or .v file")
    if len(selected) > 128:
        raise ValueError("A source bundle may contain at most 128 files")
    output = io.BytesIO()
    manifest = json.dumps({"version": 1, "entrypoint": entry.name})
    total = len(manifest.encode("utf-8"))
    seen = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mrg-project.json", manifest)
        for name in selected:
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", name):
                raise ValueError("Source paths must use letters, digits, _, -, . and /")
            if any(part in ("", ".", "..") for part in name.split("/")):
                raise ValueError("Sources must be relative to the main design's folder")
            if name.casefold() in seen:
                raise ValueError(f"Duplicate source path: {name}")
            seen.add(name.casefold())
            if Path(name).suffix not in {".py", ".v", ".vh", ".mem", ".hex"}:
                raise ValueError(f"Unsupported source file: {name}")
            path = root / name
            if any(part.is_symlink() for part in [path, *path.parents] if part != root):
                raise ValueError(f"Source symlinks are not supported: {name}")
            if not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Source must be a regular file inside {root}: {name}")
            with path.open("rb") as source:
                content = source.read(16 * 1024 * 1024 - total + 1)
            total += len(content)
            if total > 16 * 1024 * 1024:
                raise ValueError("Source bundle exceeds the 16 MiB expanded limit")
            archive.writestr(name, content)
    if output.tell() > 8 * 1024 * 1024:
        raise ValueError("Source bundle exceeds the 8 MiB upload limit")
    return output.getvalue()

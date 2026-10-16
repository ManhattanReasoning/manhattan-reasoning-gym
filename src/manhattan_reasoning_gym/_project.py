"""Create bounded cloud source bundles from a statically selected project."""

import io
import json
import zipfile

from ._discovery import discover


def source_bundle(design, sources=None, include_dir=None, *, single_file_raw=False):
    project = discover(design, sources, include_dir)
    if single_file_raw and len(project.files) == 1:
        return None
    manifest = json.dumps(
        {
            "version": 1,
            "entrypoint": project.entrypoint,
            "verilog_sources": project.verilog_sources,
        }
    )
    if len(manifest.encode()) > 4096:
        raise ValueError("Source manifest exceeds 4 KiB")
    if (
        len(manifest.encode()) + sum(map(len, project.files.values()))
        > 16 * 1024 * 1024
    ):
        raise ValueError("Source bundle exceeds the 16 MiB expanded limit")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mrg-project.json", manifest)
        for name, content in project.files.items():
            archive.writestr(name, content)
    if output.tell() > 8 * 1024 * 1024:
        raise ValueError("Source bundle exceeds the 8 MiB upload limit")
    return output.getvalue()

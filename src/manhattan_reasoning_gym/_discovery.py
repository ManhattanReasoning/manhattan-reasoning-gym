"""Static, bounded dependency discovery. Never imports or runs user code."""

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path

SUFFIXES = {".py", ".v", ".vh", ".mem", ".hex"}
MAX_BYTES = 16 * 1024 * 1024
MAX_FILES = 128
# Strip comments while preserving strings (including comment-like text in paths).
_VERILOG_TEXT = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', re.S)
_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|[A-Za-z_$][\w$]*|[^\s]')


def verilog_text(text):
    return _VERILOG_TEXT.sub(lambda m: m[0] if m[0].startswith('"') else " ", text)


def module_references(text):
    """Recognize ordinary and parameterized instances, including instance arrays."""
    tokens = _TOKEN.findall(text)
    result = set()
    for i, token in enumerate(tokens[:-2]):
        if not re.fullmatch(r"[A-Za-z_][\w$]*", token):
            continue
        j = i + 1
        if tokens[j] == "#":
            j += 1
            if tokens[j] != "(":
                continue
            depth = 0
            while j < len(tokens):
                depth += (tokens[j] == "(") - (tokens[j] == ")")
                j += 1
                if depth == 0:
                    break
        if j >= len(tokens) or not re.fullmatch(r"[A-Za-z_][\w$]*", tokens[j]):
            continue
        j += 1
        while j < len(tokens) and tokens[j] == "[":
            while j < len(tokens) and tokens[j] != "]":
                j += 1
            j += 1
        if j < len(tokens) and tokens[j] == "(":
            result.add(token)
    return result


@dataclass
class Project:
    entrypoint: str
    files: dict[str, bytes]
    verilog_sources: list[str]


def discover(design, sources=None, include_dir=None):
    entry = Path(design).absolute()
    entry = entry.parent.resolve() / entry.name
    if entry.suffix not in {".py", ".v"}:
        raise ValueError("The main design must be a .py or .v file")
    root = entry.parent.resolve()
    if include_dir is not None:
        root = Path(include_dir)
        if not root.is_absolute():
            root = entry.parent / root
        if root.is_symlink() or not root.is_dir():
            raise ValueError("include_dir must be a regular directory")
        root = root.resolve()
    cache = {}
    files = {}
    names = {}
    queue = []
    included = set()
    total = 0

    def name_for(path):
        if path.is_relative_to(root):
            return path.relative_to(root).as_posix()
        if path == entry:
            return entry.name
        raise ValueError(f"Dependency escapes the project folder: {path}")

    def read(path):
        nonlocal total
        path = Path(os.path.abspath(path))
        name_for(path)
        cursor = path
        boundary = root if path.is_relative_to(root) else entry.parent
        while cursor != boundary:
            if cursor.is_symlink():
                raise ValueError(f"Source symlinks are not supported: {path}")
            cursor = cursor.parent
        if path not in cache:
            if not path.is_file():
                raise ValueError(f"Missing source file: {path}")
            with path.open("rb") as f:
                content = f.read(MAX_BYTES - total + 1)
            total += len(content)
            if total > MAX_BYTES:
                raise ValueError(
                    "Dependency scan exceeds 16 MiB; use a smaller project directory"
                )
            cache[path] = content
        return cache[path]

    def add(path):
        path = Path(os.path.abspath(path))
        name = name_for(path)
        if (
            not re.fullmatch(r"[A-Za-z0-9_./-]+", name)
            or any(part in ("", ".", "..") for part in name.split("/"))
            or path.suffix not in SUFFIXES
        ):
            raise ValueError(f"Unsupported source path: {name}")
        previous = names.get(name.casefold())
        if previous is not None and previous != path:
            raise ValueError(f"Duplicate source path: {name}")
        if path not in files:
            if len(files) >= MAX_FILES:
                raise ValueError("A source bundle may contain at most 128 files")
            files[path] = read(path)
            names[name.casefold()] = path
            queue.append(path)

    def walk():
        count = 0
        for directory, dirs, filenames in os.walk(root, followlinks=False):
            count += len(dirs)
            if count > 4096:
                raise ValueError(
                    "Project scan exceeds 4096 entries; choose a smaller directory"
                )
            dirs[:] = sorted(
                d
                for d in dirs
                if not d.startswith(".")
                and d not in {"__pycache__", "node_modules", "build", "dist"}
                and not (Path(directory) / d).is_symlink()
            )
            for filename in sorted(filenames):
                count += 1
                if count > 4096:
                    raise ValueError(
                        "Project scan exceeds 4096 files; choose a smaller directory"
                    )
                path = Path(directory) / filename
                if not filename.startswith(".") and path.suffix in SUFFIXES:
                    yield path

    add(entry)
    # Explicit files remain a deterministic override for static-discovery edge cases.
    if sources is not None:
        for source in sources:
            if not isinstance(source, str) or any(
                p in ("", ".", "..") for p in source.split("/")
            ):
                raise ValueError("Sources must be relative paths inside the project")
            path = root / source
            if path == entry or path in files:
                raise ValueError(f"Duplicate source path: {source}")
            add(path)
    elif include_dir is not None:
        # A user-supplied directory explicitly authorizes bundling its source files.
        for path in walk():
            add(path)
    else:
        modules = None
        while queue:
            path = queue.pop(0)
            if path.suffix == ".py":
                tree = ast.parse(files[path], filename=str(path))
                package = list(path.relative_to(root).parent.parts)

                def python_module(parts):
                    candidate = root.joinpath(*parts)
                    if candidate.with_suffix(".py").is_file():
                        add(candidate.with_suffix(".py"))
                    elif (candidate / "__init__.py").is_file():
                        add(candidate / "__init__.py")
                    for i in range(1, len(parts)):
                        init = root.joinpath(*parts[:i], "__init__.py")
                        if init.is_file():
                            add(init)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            python_module(alias.name.split("."))
                    elif isinstance(node, ast.ImportFrom):
                        if node.level > len(package):
                            raise ValueError(f"Relative import escapes project: {path}")
                        base = (
                            package[: len(package) - node.level + 1]
                            if node.level
                            else []
                        )
                        parts = base + (node.module.split(".") if node.module else [])
                        python_module(parts)
                        for alias in node.names:
                            if alias.name != "*":
                                python_module(parts + [alias.name])
            elif path.suffix in {".v", ".vh"}:
                text = verilog_text(files[path].decode("utf-8"))
                if re.search(r'`include\s+(?![\s"])[^\s]+', text):
                    raise ValueError(
                        "Macro-based include requires include_dir or explicit sources"
                    )
                for target in re.findall(r'`include\s+"([^"\n]+)"', text):
                    candidates = [path.parent / target, root / target]
                    # Do not resolve symlinks before add() has a chance to reject them.
                    found = next((p for p in candidates if p.is_file()), None)
                    if not found:
                        matches = {p for p in walk() if p.name == target}
                        if len(matches) > 1:
                            raise ValueError(
                                f"Ambiguous include {target}; use a qualified path"
                            )
                        found = next(iter(matches), None)
                    if found is None:
                        raise ValueError(
                            f"Missing include {target}; supply include_dir or sources"
                        )
                    add(found)
                    included.add(Path(os.path.abspath(found)))
                # Literal memory initialization files are dependencies too.
                for target in re.findall(r'\$readmem[hb]\s*\(\s*"([^"\n]+)"', text):
                    add(root / target)
                if modules is None:
                    modules = {}
                    for candidate in walk():
                        if candidate.suffix == ".v":
                            contents = verilog_text(read(candidate).decode("utf-8"))
                            contents = re.sub(r'"(?:\\.|[^"\\])*"', '""', contents)
                            for module in re.findall(
                                r"\bmodule\s+([A-Za-z_][\w$]*)", contents
                            ):
                                modules.setdefault(module, []).append(candidate)
                for module in module_references(text):
                    candidates = modules.get(module, [])
                    if len(candidates) > 1:
                        raise ValueError(
                            f"Ambiguous module {module}; use explicit sources"
                        )
                    if candidates:
                        add(candidates[0])

    # A .v brought in textually must not also be a compilation unit.
    for path, content in files.items():
        if path.suffix in {".v", ".vh"}:
            for target in re.findall(
                r'`include\s+"([^"\n]+)"', verilog_text(content.decode("utf-8"))
            ):
                for candidate in (path.parent / target, root / target):
                    candidate = Path(os.path.abspath(candidate))
                    if candidate in files:
                        included.add(candidate)
                        break
                else:
                    matches = [p for p in files if p.name == target]
                    if len(matches) == 1:
                        included.add(matches[0])
                    elif len(matches) > 1:
                        raise ValueError(
                            f"Ambiguous include {target}; use a qualified path"
                        )
    units = sorted(
        name_for(path)
        for path in files
        if path.suffix == ".v" and path != entry and path not in included
    )
    return Project(
        name_for(entry), {name_for(p): data for p, data in files.items()}, units
    )

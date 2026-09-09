"""Read-only MCP access to the current DeepTrack2 repository checkout.

The server deliberately works from repository files instead of a generated
API catalogue.  This keeps its answers aligned with the checkout that starts
the server and avoids importing tutorial notebooks or executing user input.
The MCP dependency is optional; importing this module without it is safe, but
``main`` will report how to install the optional extra.
"""

from __future__ import annotations

import ast
import difflib
import importlib
import inspect
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple

try:
    from mcp.server import MCPServer
except ImportError:  # pragma: no cover - exercised in minimal installs
    MCPServer = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 256
MAX_RESULTS = 20
MAX_COMPONENTS = 160
MAX_EXCERPT_CHARS = 560
MAX_DOCSTRING_CHARS = 8_000
MAX_BASE_CLASSES = 16
MAX_SOURCE_LINES = 240
MAX_SOURCE_CHARS = 32_000
MAX_INDEX_FILE_BYTES = 16 * 1024 * 1024
MAX_CONTEXT_CHARS = 4_000
MAX_RELATED_SYMBOLS = 16
MAX_RELATED_CELLS = 2
MAX_INHERITED_PARAMETERS = 64
_DIVERSITY_PENALTY_FRACTION = 0.15
_COVERAGE_WEIGHT = 36
_FULL_COVERAGE_BONUS = 30
_EXACT_PUBLIC_SYMBOL_BONUS = 60

_KIND_PRIORS = {
    "api_symbol": 4,
    "tutorial_code": 13,
    "tutorial_markdown": 7,
    "documentation": 2,
    "python_source": 0,
    "test": -9,
}
_MCP_QUERY_TERMS = frozenset(
    {"mcp", "model", "context", "protocol", "inspector", "agent"}
)

_SUPPORTED_SUFFIXES = frozenset({".py", ".md", ".rst", ".ipynb"})
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".venv312",
        "__pycache__",
        "datasets",
        "lightning_logs",
    }
)

# These are family-to-module rules, not a duplicated component catalogue.
# The component names are discovered from the checkout's AST and, when
# available, the modules' __all__ values.
_CATEGORY_MODULES: dict[str, tuple[str, ...]] = {
    "scatterers": ("deeptrack.optical.scatterers",),
    "optics": ("deeptrack.optical.optics",),
    "aberrations": ("deeptrack.optical.aberrations",),
    "noise/features": ("deeptrack.optical.noises", "deeptrack.features"),
    "sequences": ("deeptrack.sequences",),
    "pytorch": ("deeptrack.pytorch",),
}
_CATEGORY_ALIASES = {
    "noise": "noise/features",
    "noises": "noise/features",
    "feature": "noise/features",
    "features": "noise/features",
    "pytorch integration": "pytorch",
    "torch": "pytorch",
}
_SYMBOL_PATTERN = re.compile(
    r"(?:deeptrack\.)?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z"
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*")


# ``mcp dev`` may execute this file without registering its module in
# ``sys.modules``.  NamedTuple keeps these records immutable without the
# postponed-annotation lookup that Python 3.14's dataclass decorator performs.
class SearchRecord(NamedTuple):
    """One searchable repository item."""

    kind: str
    title: str
    path: str
    text: str
    line: int | None = None
    notebook: str | None = None
    cell: int | None = None
    cell_type: str | None = None
    preceding_markdown_cell: int | None = None
    context: str = ""
    referenced_symbols: tuple[str, ...] = ()
    called_symbols: tuple[str, ...] = ()
    referenced_modules: tuple[str, ...] = ()


class SymbolInfo(NamedTuple):
    """Static or runtime information about a public package symbol."""

    name: str
    qualified: str
    module: str
    kind: str
    path: str
    line: int | None
    signature: str
    docstring: str
    bases: tuple[str, ...]


def _clip(value: str, length: int) -> str:
    """Return a bounded string without making output depend on encoding."""

    value = value or ""
    if len(value) <= length:
        return value
    return value[: max(0, length - 1)].rstrip() + "…"


def _path_inside(root: Path, candidate: Path) -> bool:
    """Return whether ``candidate`` resolves below ``root``."""

    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _repository_markers(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "deeptrack").is_dir()
        and (path / "tutorials").is_dir()
        and (
            (path / "setup.py").is_file()
            or (path / "pyproject.toml").is_file()
        )
    )


def resolve_repository_root(start: str | Path | None = None) -> Path:
    """Find the DeepTrack2 checkout containing this server.

    The module location is tried before the process working directory, so a
    console script continues to work when launched by an IDE from elsewhere.
    ``DEEPTRACK_REPOSITORY`` is an explicit override for editable or unusual
    installations.
    """

    candidates: list[Path] = []
    override = os.environ.get("DEEPTRACK_REPOSITORY")
    if override:
        candidates.append(Path(override))
    if start is not None:
        candidates.append(Path(start))
    candidates.extend([Path(__file__).resolve().parent, Path.cwd()])

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            current = candidate.expanduser().resolve()
        except OSError:
            continue
        if current.is_file():
            current = current.parent
        for parent in (current, *current.parents):
            if parent in seen:
                continue
            seen.add(parent)
            if _repository_markers(parent):
                return parent

    raise RuntimeError(
        "Could not locate a DeepTrack2 repository checkout. Start the server "
        "from a checkout, install it editable, or set DEEPTRACK_REPOSITORY."
    )


def _module_name(root: Path, path: Path) -> str | None:
    """Convert a package source path to its importable module name."""

    try:
        relative = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    if not relative.parts or relative.parts[0] != "deeptrack":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or "deeptrack"


def _relative_path(root: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _iter_repository_files(root: Path) -> Iterator[Path]:
    """Yield supported files in stable order, excluding generated data."""

    files: set[Path] = set()
    scan_roots = [
        root / name for name in ("deeptrack", "tutorials", "tests", "docs")
    ]
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        for path in scan_root.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() not in _SUPPORTED_SUFFIXES
            ):
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if _path_inside(root, path):
                files.add(path.resolve())

    # The current checkout keeps its primary documentation at the root.  Also
    # include any future root-level README or reStructuredText documents.
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in {".md", ".rst"}:
            if _path_inside(root, path):
                files.add(path.resolve())

    for path in sorted(files, key=lambda item: item.as_posix()):
        yield path


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_INDEX_FILE_BYTES:
            logger.warning("Skipping oversized index file: %s", path)
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        logger.warning("Unable to read %s: %s", path, error)
        return None


def _document_title(text: str, path: Path) -> str:
    for line in text.splitlines()[:40]:
        match = re.match(r"^\s{0,3}#+\s+(.+?)\s*#*\s*$", line)
        if match:
            return _clip(match.group(1), 180)
    return path.stem


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return "?"


def _argument_text(arg: ast.arg, default: ast.AST | None = None) -> str:
    result = arg.arg
    if arg.annotation is not None:
        result += ": " + _unparse(arg.annotation)
    if default is not None:
        result += " = " + _unparse(default)
    return result


def _static_arguments(
    arguments: ast.arguments,
    drop_first: bool = False,
) -> str:
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.AST | None] = [None] * (
        len(positional) - len(arguments.defaults)
    ) + list(arguments.defaults)
    if drop_first and positional:
        positional = positional[1:]
        defaults = defaults[1:]

    parts = [
        _argument_text(argument, default)
        for argument, default in zip(positional, defaults)
    ]
    if arguments.posonlyargs and not drop_first:
        parts.insert(len(arguments.posonlyargs), "/")
    if arguments.vararg is not None:
        parts.append("*" + _argument_text(arguments.vararg))
    elif arguments.kwonlyargs:
        parts.append("*")
    parts.extend(
        _argument_text(argument, default)
        for argument, default in zip(
            arguments.kwonlyargs,
            arguments.kw_defaults,
        )
    )
    if arguments.kwarg is not None:
        parts.append("**" + _argument_text(arguments.kwarg))
    return _clip("(" + ", ".join(parts) + ")", 1_000)


def _static_signature(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    if isinstance(node, ast.ClassDef):
        initializer = next(
            (
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "__init__"
            ),
            None,
        )
        if initializer is None:
            return "(...)"
        return _static_arguments(initializer.args, drop_first=True)
    return _static_arguments(node.args, drop_first=False)


def _summary(docstring: str) -> str:
    paragraphs = [
        part.strip() for part in docstring.split("\n\n") if part.strip()
    ]
    return _clip(" ".join(paragraphs[0].split()) if paragraphs else "", 320)


def _parse_symbols(root: Path, path: Path, text: str) -> list[SymbolInfo]:
    module = _module_name(root, path)
    relative = _relative_path(root, path)
    if module is None or relative is None:
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        logger.warning("Unable to parse %s: %s", path, error)
        return []

    symbols: list[SymbolInfo] = []
    for node in tree.body:
        if not isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        if node.name.startswith("_"):
            continue
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        bases = (
            tuple(_clip(_unparse(base), 300) for base in node.bases)[
                :MAX_BASE_CLASSES
            ]
            if isinstance(node, ast.ClassDef)
            else ()
        )
        docstring = ast.get_docstring(node, clean=False) or ""
        symbols.append(
            SymbolInfo(
                name=node.name,
                qualified=f"{module}.{node.name}",
                module=module,
                kind=kind,
                path=relative,
                line=getattr(node, "lineno", None),
                signature=_static_signature(node),
                docstring=docstring,
                bases=bases,
            )
        )
    return symbols


def _notebook_records(root: Path, path: Path, text: str) -> list[SearchRecord]:
    relative = _relative_path(root, path)
    if relative is None:
        return []
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as error:
        logger.warning("Unable to parse notebook %s: %s", path, error)
        return []
    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        return []

    title = _document_title(
        "\n".join(
            "".join(cell.get("source", []))
            for cell in cells
            if isinstance(cell, dict) and cell.get("cell_type") == "markdown"
        ),
        path,
    )
    records: list[SearchRecord] = []
    preceding_markdown_cell: int | None = None
    preceding_markdown_text = ""
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            preceding_markdown_cell = None
            preceding_markdown_text = ""
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in {"markdown", "code", "raw"}:
            preceding_markdown_cell = None
            preceding_markdown_text = ""
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(str(part) for part in source)
        if not isinstance(source, str) or not source.strip():
            preceding_markdown_cell = None
            preceding_markdown_text = ""
            continue
        records.append(
            SearchRecord(
                kind=f"tutorial_{cell_type}",
                title=title,
                path=f"{relative}#cell-{index}",
                text=source,
                notebook=relative,
                cell=index,
                cell_type=cell_type,
                preceding_markdown_cell=(
                    preceding_markdown_cell if cell_type == "code" else None
                ),
                context=(
                    _clip(preceding_markdown_text, MAX_CONTEXT_CHARS)
                    if cell_type == "code"
                    else ""
                ),
            )
        )
        if cell_type == "markdown":
            preceding_markdown_cell = index
            preceding_markdown_text = source
        else:
            preceding_markdown_cell = None
            preceding_markdown_text = ""
    return records


def _normalise_symbol(value: str) -> str:
    value = value.strip()
    if value.startswith("dt."):
        value = "deeptrack." + value[3:]
    if not value.startswith("deeptrack."):
        return value
    return value


def _query_tokens(query: str) -> list[str]:
    return list(
        dict.fromkeys(
            token.casefold() for token in _TOKEN_PATTERN.findall(query)
        )
    )


def _is_uninformative_signature(signature: str) -> bool:
    """Return whether a signature contains no useful parameter detail."""

    return bool(
        re.fullmatch(
            r"\(\.\.\.\)(?:\s*->\s*[^\s]+)?",
            signature.strip(),
        )
    )


def _class_constructor(
    value: Any,
) -> tuple[Any, inspect.Signature] | None:
    """Find the first useful constructor in a class's MRO."""

    if not inspect.isclass(value):
        return None
    for owner in inspect.getmro(value):
        if owner is object:
            continue
        for attribute_name in ("__init__", "__new__"):
            constructor = owner.__dict__.get(attribute_name)
            if constructor is None:
                continue
            try:
                signature = inspect.signature(constructor)
                parameters = list(signature.parameters.values())
                if parameters and parameters[0].name in {"self", "cls"}:
                    signature = signature.replace(parameters=parameters[1:])
                rendered = str(signature)
            except (TypeError, ValueError):
                continue
            if _is_uninformative_signature(rendered):
                continue
            return owner, signature
    return None


def _parameter_details(
    signature: inspect.Signature,
    source: str,
) -> list[dict[str, Any]]:
    """Serialize constructor parameters without leaking unbounded reprs."""

    parameters: list[dict[str, Any]] = []
    for parameter in list(signature.parameters.values())[
        :MAX_INHERITED_PARAMETERS
    ]:
        item: dict[str, Any] = {
            "name": parameter.name,
            "kind": parameter.kind.name.lower(),
            "required": (
                parameter.default is inspect.Parameter.empty
                and parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            ),
            "source": source,
        }
        if parameter.annotation is not inspect.Parameter.empty:
            item["annotation"] = _clip(str(parameter.annotation), 320)
        if parameter.default is not inspect.Parameter.empty:
            item["default"] = _clip(repr(parameter.default), 320)
        parameters.append(item)
    return parameters


def _extract_tutorial_references(
    text: str,
    symbols: Iterator[SymbolInfo],
    module_names: Iterator[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Find statically recognizable symbol calls and module references."""

    symbol_list = sorted(
        symbols,
        key=lambda item: (len(item.name), item.name, item.qualified),
        reverse=True,
    )
    symbol_by_name: dict[str, list[str]] = {}
    for symbol in symbol_list:
        symbol_by_name.setdefault(symbol.name, []).append(symbol.qualified)
    symbol_names = sorted(symbol_by_name, key=len, reverse=True)
    if symbol_names:
        symbol_alternation = "|".join(re.escape(name) for name in symbol_names)
        reference_pattern = re.compile(
            rf"(?<![\w.])(?:[A-Za-z_]\w*\.)*"
            rf"(?P<name>{symbol_alternation})(?!\w)"
        )
        call_pattern = re.compile(
            rf"(?<![\w.])(?:[A-Za-z_]\w*\.)*"
            rf"(?P<name>{symbol_alternation})\s*\("
        )
        referenced_names = {
            match.group("name") for match in reference_pattern.finditer(text)
        }
        called_names = {
            match.group("name") for match in call_pattern.finditer(text)
        }
    else:
        referenced_names = set()
        called_names = set()

    referenced_symbols = [
        qualified
        for symbol in symbol_list
        if symbol.name in referenced_names
        for qualified in symbol_by_name[symbol.name]
    ]
    called_symbols = [
        qualified
        for symbol in symbol_list
        if symbol.name in called_names
        for qualified in symbol_by_name[symbol.name]
    ]

    module_list = sorted(
        (module for module in module_names if module != "deeptrack"),
        key=len,
        reverse=True,
    )
    module_by_leaf: dict[str, list[str]] = {}
    for module in module_list:
        module_by_leaf.setdefault(module.rsplit(".", 1)[-1], []).append(module)
    if module_list:
        module_alternation = "|".join(
            re.escape(module) for module in module_list
        )
        full_module_pattern = re.compile(
            rf"(?<![\w.])(?P<module>{module_alternation})(?![\w.])"
        )
        referenced_modules = {
            match.group("module")
            for match in full_module_pattern.finditer(text)
        }
    else:
        referenced_modules = set()
    if module_by_leaf:
        leaf_alternation = "|".join(
            re.escape(leaf)
            for leaf in sorted(module_by_leaf, key=len, reverse=True)
        )
        member_pattern = re.compile(
            rf"(?<![\w.])(?P<leaf>{leaf_alternation})\s*\."
        )
        for match in member_pattern.finditer(text):
            referenced_modules.update(module_by_leaf[match.group("leaf")])

        from_pattern = re.compile(
            r"(?m)^\s*from\s+(?P<parent>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"
            r"\s+import\s+(?P<imports>[^\n#]+)"
        )
        for match in from_pattern.finditer(text):
            parent = match.group("parent")
            imported = set(_TOKEN_PATTERN.findall(match.group("imports")))
            for leaf in imported:
                for module in module_by_leaf.get(leaf, ()):
                    if module.rsplit(".", 1)[0] == parent:
                        referenced_modules.add(module)

    return (
        tuple(dict.fromkeys(referenced_symbols)),
        tuple(dict.fromkeys(called_symbols)),
        tuple(sorted(referenced_modules)),
    )


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"error": {"code": code, "message": message}}
    result["error"].update(details)
    return result


class RepositoryKnowledge:
    """Read-only, lazily refreshed knowledge of one DeepTrack2 checkout."""

    def __init__(self, repository_root: str | Path | None = None) -> None:
        self.root = resolve_repository_root(repository_root)
        self._signature: tuple[tuple[str, int, int], ...] | None = None
        self._records: tuple[SearchRecord, ...] = ()
        self._symbols: tuple[SymbolInfo, ...] = ()
        self._module_names: frozenset[str] = frozenset()

    def _refresh(self) -> None:
        files = list(_iter_repository_files(self.root))
        signature: list[tuple[str, int, int]] = []
        for path in files:
            relative = _relative_path(self.root, path)
            try:
                stat = path.stat()
            except OSError:
                continue
            if relative is not None:
                signature.append((relative, stat.st_mtime_ns, stat.st_size))
        current_signature = tuple(signature)
        if current_signature == self._signature:
            return

        records: list[SearchRecord] = []
        symbols: list[SymbolInfo] = []
        module_names: set[str] = set()
        for path in files:
            relative = _relative_path(self.root, path)
            text = _read_text(path)
            if relative is None or text is None:
                continue
            if path.suffix.lower() == ".ipynb":
                records.extend(_notebook_records(self.root, path, text))
                continue

            if relative.startswith("tests/"):
                kind = "test"
            elif path.suffix.lower() == ".py":
                kind = "python_source"
            else:
                kind = "documentation"
            records.append(
                SearchRecord(
                    kind=kind,
                    title=_document_title(text, path),
                    path=relative,
                    text=text,
                )
            )
            if path.suffix.lower() == ".py":
                module = _module_name(self.root, path)
                if module is not None:
                    module_names.add(module)
                    parsed = _parse_symbols(self.root, path, text)
                    symbols.extend(parsed)
                    records.extend(
                        SearchRecord(
                            kind="api_symbol",
                            title=symbol.qualified,
                            path=symbol.path,
                            line=symbol.line,
                            text=(
                                f"{symbol.qualified}\n{symbol.docstring}\n"
                                f"{' '.join(symbol.bases)}"
                            ),
                        )
                        for symbol in parsed
                    )

        annotated_records: list[SearchRecord] = []
        for record in records:
            if record.notebook is None:
                annotated_records.append(record)
                continue
            references, calls, modules = _extract_tutorial_references(
                record.text,
                iter(symbols),
                iter(module_names),
            )
            annotated_records.append(
                record._replace(
                    referenced_symbols=references,
                    called_symbols=calls,
                    referenced_modules=modules,
                )
            )

        self._records = tuple(annotated_records)
        self._symbols = tuple(
            sorted(
                symbols,
                key=lambda symbol: (symbol.name.casefold(), symbol.qualified),
            )
        )
        self._module_names = frozenset(module_names)
        self._signature = current_signature

    def _safe_file(self, relative: str) -> Path | None:
        candidate = (self.root / relative).resolve()
        if not _path_inside(self.root, candidate) or not candidate.is_file():
            return None
        return candidate

    def _public_symbols(self) -> tuple[SymbolInfo, ...]:
        """Return discovered DeepTrack symbols, excluding this server."""

        return tuple(
            symbol
            for symbol in self._symbols
            if symbol.module != "deeptrack.mcp_server"
        )

    def _matching_public_symbols(
        self,
        query: str,
        case_sensitive: bool = False,
    ) -> tuple[SymbolInfo, ...]:
        raw_tokens = _TOKEN_PATTERN.findall(query)
        query_tokens = {
            token if case_sensitive else token.casefold()
            for token in raw_tokens
        }
        matches: list[SymbolInfo] = []
        seen: set[str] = set()
        for symbol in self._public_symbols():
            name = symbol.name if case_sensitive else symbol.name.casefold()
            if name not in query_tokens or symbol.qualified in seen:
                continue
            seen.add(symbol.qualified)
            matches.append(symbol)
        return tuple(matches[:MAX_RELATED_SYMBOLS])

    def _matching_modules(
        self,
        query: str,
        explicit_only: bool = False,
    ) -> tuple[str, ...]:
        query_casefold = query.casefold()
        raw_tokens = _TOKEN_PATTERN.findall(query)
        matches: list[str] = []
        for module in sorted(self._module_names, key=len, reverse=True):
            if module == "deeptrack":
                continue
            leaf = module.rsplit(".", 1)[-1]
            explicit = bool(
                re.search(
                    rf"(?<![\w.]){re.escape(module.casefold())}" rf"(?![\w.])",
                    query_casefold,
                )
            )
            bare_module = leaf in raw_tokens
            if explicit or (not explicit_only and bare_module):
                matches.append(module)

        # A fully qualified module also contains its parent package names.
        # Keep the most specific discovered module in that case.
        return tuple(
            module
            for module in matches
            if not any(
                other != module and other.startswith(module + ".")
                for other in matches
            )
        )

    @staticmethod
    def _search_text(record: SearchRecord) -> str:
        if record.context:
            return record.text + "\n" + record.context
        return record.text

    def _score(
        self,
        record: SearchRecord,
        query: str,
        tokens: list[str],
        public_targets: tuple[SymbolInfo, ...] | None = None,
    ) -> tuple[int, list[str], dict[str, Any]]:
        content = self._search_text(record).casefold()
        metadata = f"{record.title} {record.path}".casefold()
        matched: list[str] = []
        score = 0
        for token in tokens:
            content_count = len(
                re.findall(rf"\b{re.escape(token)}\b", content)
            )
            if content_count:
                matched.append(token)
                score += min(content_count, 8) * 2
            if token in record.title.casefold():
                if token not in matched:
                    matched.append(token)
                score += 8
            if token in record.path.casefold():
                if token not in matched:
                    matched.append(token)
                score += 5
        phrase = " ".join(query.casefold().split())
        if phrase and phrase in content:
            score += 5
        if phrase and phrase in metadata:
            score += 3

        matched_set = set(matched)
        coverage = len(matched_set) / max(1, len(tokens))
        score += round(_COVERAGE_WEIGHT * coverage)
        if coverage == 1:
            score += _FULL_COVERAGE_BONUS

        kind_prior = _KIND_PRIORS.get(record.kind, 0)
        score += kind_prior
        signals: dict[str, Any] = {
            "coverage": round(coverage, 2),
        }
        if kind_prior:
            signals["kind_prior"] = kind_prior

        if public_targets is None:
            public_targets = self._matching_public_symbols(query)
        exact_public = tuple(
            target
            for target in public_targets
            if record.kind == "api_symbol"
            and record.title.casefold() == target.qualified.casefold()
        )
        if exact_public:
            score += (
                _EXACT_PUBLIC_SYMBOL_BONUS + max(0, len(exact_public) - 1) * 8
            )
            signals["exact_public_symbols"] = [
                target.qualified for target in exact_public
            ][:MAX_RELATED_SYMBOLS]

        called_public = tuple(
            target
            for target in public_targets
            if target.qualified in record.called_symbols
        )
        referenced_public = tuple(
            target
            for target in public_targets
            if target.qualified in record.referenced_symbols
        )
        if called_public and record.kind == "tutorial_code":
            score += 20
            signals["called_public_symbols"] = [
                target.qualified for target in called_public
            ][:MAX_RELATED_SYMBOLS]
        elif referenced_public:
            score += 8
            signals["referenced_public_symbols"] = [
                target.qualified for target in referenced_public
            ][:MAX_RELATED_SYMBOLS]

        query_casefold = query.casefold().strip()
        exact_query_symbols = tuple(
            target
            for target in public_targets
            if query_casefold
            in {target.name.casefold(), target.qualified.casefold()}
        )
        if exact_query_symbols and exact_public:
            score += 18
            signals["exact_symbol_query"] = True

        ordinary_query = not bool(matched_set.intersection(_MCP_QUERY_TERMS))
        if ordinary_query:
            path = record.path.casefold()
            if path == "mcp.md":
                score -= 32
                signals["repository_noise_penalty"] = 32
            elif path == "tests/test_mcp_server.py":
                score -= 32
                signals["repository_noise_penalty"] = 32

        return score, matched, signals

    def _excerpt(self, record: SearchRecord, tokens: list[str]) -> str:
        text = record.text.strip()
        if len(text) <= MAX_EXCERPT_CHARS:
            return text
        lower = text.casefold()
        positions = [
            lower.find(token) for token in tokens if lower.find(token) >= 0
        ]
        start_at = min(positions) if positions else 0
        half = MAX_EXCERPT_CHARS // 2
        start = max(0, start_at - half)
        end = min(len(text), start + MAX_EXCERPT_CHARS)
        if end - start < MAX_EXCERPT_CHARS:
            start = max(0, end - MAX_EXCERPT_CHARS)
        excerpt = text[start:end].strip()
        if start:
            excerpt = "…" + excerpt
        if end < len(text):
            excerpt += "…"
        return _clip(excerpt, MAX_EXCERPT_CHARS)

    def _prepare_query(
        self,
        query: str,
        limit: int,
    ) -> tuple[str, list[str], int] | dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return _error(
                "invalid_query",
                "query must contain searchable text",
            )
        query = query.strip()
        if len(query) > MAX_QUERY_CHARS:
            query = query[:MAX_QUERY_CHARS]
        tokens = _query_tokens(query)
        if not tokens:
            return _error(
                "invalid_query",
                "query must contain searchable text",
            )
        try:
            effective_limit = max(1, min(int(limit), MAX_RESULTS))
        except (TypeError, ValueError):
            effective_limit = 8

        return query, tokens, effective_limit

    def _score_example(
        self,
        record: SearchRecord,
        query: str,
        tokens: list[str],
        public_targets: tuple[SymbolInfo, ...],
        module_targets: tuple[str, ...],
    ) -> tuple[int, list[str], dict[str, Any]]:
        score, matched, signals = self._score(
            record,
            query,
            tokens,
            public_targets,
        )
        called = set(record.called_symbols)
        referenced = set(record.referenced_symbols)
        if public_targets:
            target_names = {target.qualified for target in public_targets}
            called_targets = sorted(called.intersection(target_names))
            referenced_targets = sorted(referenced.intersection(target_names))
            if called_targets:
                score += 78
                signals["example_public_symbols"] = called_targets[
                    :MAX_RELATED_SYMBOLS
                ]
            elif referenced_targets:
                score += 24
                signals["example_public_symbols"] = referenced_targets[
                    :MAX_RELATED_SYMBOLS
                ]

        if module_targets:
            module_references = sorted(
                set(module_targets).intersection(record.referenced_modules)
            )
            module_calls = sorted(
                symbol
                for symbol in called
                if any(
                    symbol.startswith(module + ".")
                    for module in module_targets
                )
            )
            if module_references:
                score += 58
                signals["example_modules"] = module_references[
                    :MAX_RELATED_SYMBOLS
                ]
            if module_calls:
                score += 26
                signals["example_module_symbols"] = module_calls[
                    :MAX_RELATED_SYMBOLS
                ]

            target_leaves = {
                module.rsplit(".", 1)[-1].casefold()
                for module in module_targets
            }
            conflicting_calls = sorted(
                symbol
                for symbol in called
                if symbol.rsplit(".", 1)[-1].casefold() in target_leaves
                and not any(
                    symbol.startswith(module + ".")
                    for module in module_targets
                )
            )
            if conflicting_calls:
                score -= 46
                signals["symbol_name_conflict_penalty"] = 46

        return score, matched, signals

    def _rank_records(
        self,
        query: str,
        limit: int,
        predicate: Callable[[SearchRecord], bool] | None = None,
        content_only: bool = False,
        example_targets: (
            tuple[tuple[SymbolInfo, ...], tuple[str, ...]] | None
        ) = None,
    ) -> (
        tuple[
            str,
            list[str],
            int,
            list[tuple[int, SearchRecord, list[str], dict[str, Any]]],
        ]
        | dict[str, Any]
    ):
        prepared = self._prepare_query(query, limit)
        if isinstance(prepared, dict):
            return prepared
        query, tokens, effective_limit = prepared

        public_targets = self._matching_public_symbols(query)
        hits: list[tuple[int, SearchRecord, list[str], dict[str, Any]]] = []
        for record in self._records:
            if predicate is not None and not predicate(record):
                continue
            if content_only and not any(
                re.search(
                    rf"\b{re.escape(token)}\b",
                    self._search_text(record).casefold(),
                )
                for token in tokens
            ):
                continue
            if example_targets is None:
                score, matched, signals = self._score(
                    record,
                    query,
                    tokens,
                    public_targets,
                )
            else:
                score, matched, signals = self._score_example(
                    record,
                    query,
                    tokens,
                    *example_targets,
                )
            if score:
                hits.append((score, record, matched, signals))
        hits.sort(
            key=lambda item: (
                -item[0],
                item[1].path,
                item[1].line or item[1].cell or 0,
            )
        )
        return query, tokens, effective_limit, hits

    def _result_item(
        self,
        score: int,
        record: SearchRecord,
        matched: list[str],
        signals: dict[str, Any],
        tokens: list[str],
    ) -> dict[str, Any]:
        relevance: dict[str, Any] = {
            "score": score,
            "matched_terms": matched,
        }
        relevance.update(signals)
        result: dict[str, Any] = {
            "kind": record.kind,
            "title": record.title,
            "path": record.path,
            "excerpt": self._excerpt(record, tokens),
            "relevance": relevance,
        }
        if record.line is not None:
            result["line"] = record.line
        if record.notebook is not None:
            result["notebook"] = record.notebook
        if record.cell is not None:
            result["cell"] = record.cell
        return result

    def _search(
        self,
        query: str,
        limit: int,
        predicate: Callable[[SearchRecord], bool] | None = None,
        content_only: bool = False,
    ) -> dict[str, Any]:
        ranked = self._rank_records(
            query,
            limit,
            predicate=predicate,
            content_only=content_only,
        )
        if isinstance(ranked, dict):
            return ranked
        query, tokens, effective_limit, hits = ranked

        results: list[dict[str, Any]] = []
        for score, record, matched, signals in hits[:effective_limit]:
            results.append(
                self._result_item(
                    score,
                    record,
                    matched,
                    signals,
                    tokens,
                )
            )
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "limit": effective_limit,
        }

    def search(self, query: str, limit: int = 8) -> dict[str, Any]:
        self._refresh()
        return self._search(query, limit)

    def _example_targets(
        self,
        query: str,
    ) -> tuple[tuple[SymbolInfo, ...], tuple[str, ...], str]:
        exact_public = self._matching_public_symbols(
            query,
            case_sensitive=True,
        )
        explicit_modules = self._matching_modules(query, explicit_only=True)
        if exact_public and not explicit_modules:
            return exact_public, (), "public_symbol"
        if explicit_modules:
            return (), explicit_modules, "module"

        bare_modules = self._matching_modules(query)
        if bare_modules and not exact_public:
            return (), bare_modules, "module"
        if exact_public:
            return exact_public, (), "public_symbol"

        fallback_public = self._matching_public_symbols(query)
        return (
            fallback_public,
            (),
            "public_symbol" if fallback_public else "concept",
        )

    def _example_group_key(
        self,
        record: SearchRecord,
        cell_map: dict[tuple[str, int], SearchRecord],
    ) -> tuple[str | None, int | None]:
        if record.notebook is None or record.cell is None:
            return record.notebook, record.cell
        if (
            record.kind == "tutorial_code"
            and record.preceding_markdown_cell is not None
        ):
            return record.notebook, record.cell
        if record.kind == "tutorial_markdown":
            following = cell_map.get((record.notebook, record.cell + 1))
            if (
                following is not None
                and following.kind == "tutorial_code"
                and following.preceding_markdown_cell == record.cell
            ):
                return record.notebook, following.cell
        return record.notebook, record.cell

    def _select_diverse_examples(
        self,
        representatives: list[
            tuple[int, SearchRecord, list[str], dict[str, Any]]
        ],
        limit: int,
    ) -> list[tuple[int, SearchRecord, list[str], dict[str, Any], int]]:
        remaining = list(representatives)
        selected: list[
            tuple[int, SearchRecord, list[str], dict[str, Any], int]
        ] = []
        notebook_counts: dict[str, int] = {}

        def selection_multiplier(record: SearchRecord) -> float:
            count = notebook_counts.get(record.notebook or "", 0)
            return max(
                0.05,
                1 - _DIVERSITY_PENALTY_FRACTION * count * count,
            )

        while remaining and len(selected) < limit:
            choice_index, choice = max(
                enumerate(remaining),
                key=lambda item: (
                    item[1][0] * selection_multiplier(item[1][1]),
                    item[1][0],
                    1 if item[1][1].kind == "tutorial_code" else 0,
                    -(item[1][1].cell or 0),
                    -item[0],
                ),
            )
            del remaining[choice_index]
            notebook = choice[1].notebook or ""
            penalty = round(
                choice[0]
                * _DIVERSITY_PENALTY_FRACTION
                * notebook_counts.get(notebook, 0) ** 2
            )
            notebook_counts[notebook] = notebook_counts.get(notebook, 0) + 1
            selected.append((*choice, penalty))
        return selected

    def _related_tutorial_cells(
        self,
        record: SearchRecord,
        tokens: list[str],
        cell_map: dict[tuple[str, int], SearchRecord],
    ) -> list[dict[str, Any]]:
        if (
            record.notebook is None
            or record.cell is None
            or record.kind != "tutorial_code"
            or record.preceding_markdown_cell is None
        ):
            return []
        preceding = cell_map.get(
            (record.notebook, record.preceding_markdown_cell)
        )
        if preceding is None or preceding.kind != "tutorial_markdown":
            return []
        return [
            {
                "kind": preceding.kind,
                "title": preceding.title,
                "path": preceding.path,
                "cell": preceding.cell,
                "excerpt": self._excerpt(preceding, tokens),
                "role": "preceding_explanation",
            }
        ][:MAX_RELATED_CELLS]

    def find_examples(self, query: str, limit: int = 5) -> dict[str, Any]:
        self._refresh()
        public_targets, module_targets, target_mode = self._example_targets(
            query
        )
        ranked = self._rank_records(
            query,
            limit,
            predicate=lambda record: record.notebook is not None,
            content_only=True,
            example_targets=(public_targets, module_targets),
        )
        if isinstance(ranked, dict):
            return ranked
        query, tokens, effective_limit, hits = ranked
        cell_map = {
            (record.notebook, record.cell): record
            for record in self._records
            if record.notebook is not None and record.cell is not None
        }
        grouped: dict[
            tuple[str | None, int | None],
            list[tuple[int, SearchRecord, list[str], dict[str, Any]]],
        ] = {}
        for hit in hits:
            grouped.setdefault(
                self._example_group_key(hit[1], cell_map),
                [],
            ).append(hit)

        representatives: list[
            tuple[int, SearchRecord, list[str], dict[str, Any]]
        ] = []
        for group in grouped.values():
            representatives.append(
                max(
                    group,
                    key=lambda item: (
                        item[0]
                        + (10 if item[1].kind == "tutorial_code" else 0),
                        item[0],
                        1 if item[1].called_symbols else 0,
                        -(item[1].cell or 0),
                    ),
                )
            )
        representatives.sort(
            key=lambda item: (
                -item[0],
                item[1].path,
                item[1].cell or 0,
            )
        )
        selected = self._select_diverse_examples(
            representatives,
            effective_limit,
        )

        results: list[dict[str, Any]] = []
        for score, record, matched, signals, penalty in selected:
            item = self._result_item(
                score,
                record,
                matched,
                signals,
                tokens,
            )
            if penalty:
                item["relevance"]["diversity_penalty"] = penalty
                item["relevance"]["selection_score"] = score - penalty
            related = self._related_tutorial_cells(
                record,
                tokens,
                cell_map,
            )
            if related:
                item["related_cells"] = related
            results.append(item)

        result = {
            "query": query,
            "results": results,
            "count": len(results),
            "limit": effective_limit,
            "examples_only": True,
            "target_mode": target_mode,
        }
        return result

    def _runtime_object(self, requested: str) -> Any | None:
        normalized = _normalise_symbol(requested)
        if not _SYMBOL_PATTERN.fullmatch(normalized):
            return None
        if normalized == "deeptrack":
            return None
        parts = normalized.split(".")
        if parts[0] == "deeptrack":
            parts = parts[1:]
        if not parts:
            return None

        # Import only known package modules.  This prevents a caller from
        # turning a symbol lookup into an arbitrary import.
        for split in range(len(parts), 0, -1):
            module = "deeptrack." + ".".join(parts[:split])
            if module not in self._module_names:
                continue
            try:
                value = importlib.import_module(module)
                for attribute in parts[split:]:
                    value = getattr(value, attribute)
                return value
            except Exception:  # pragma: no cover
                # Optional package failures vary by installed dependencies.
                return None
        try:
            value = importlib.import_module("deeptrack")
            for attribute in parts:
                value = getattr(value, attribute)
            return value
        except Exception:  # pragma: no cover - optional package failures vary
            return None

    def _runtime_info(
        self,
        value: Any,
        fallback: SymbolInfo | None,
    ) -> SymbolInfo | None:
        module = getattr(value, "__module__", "")
        qualname = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", ""),
        )
        if not module.startswith("deeptrack") or not qualname:
            return fallback
        source = None
        try:
            source = inspect.getsourcefile(value)
        except (OSError, TypeError):
            source = None
        relative = _relative_path(self.root, Path(source)) if source else None
        if relative is None and fallback is not None:
            relative = fallback.path
        if relative is None or self._safe_file(relative) is None:
            return fallback
        try:
            signature = str(inspect.signature(value))
        except (TypeError, ValueError):
            signature = fallback.signature if fallback else "(...)"
        constructor = _class_constructor(value)
        if (
            inspect.isclass(value)
            and _is_uninformative_signature(signature)
            and constructor is not None
        ):
            signature = str(constructor[1])
        try:
            docstring = inspect.getdoc(value) or ""
        except (AttributeError, TypeError):
            docstring = fallback.docstring if fallback else ""
        try:
            line = inspect.getsourcelines(value)[1]
        except (OSError, TypeError):
            line = fallback.line if fallback else None
        if inspect.isclass(value):
            bases = tuple(
                _clip(f"{base.__module__}.{base.__qualname__}", 300)
                for base in value.__bases__
                if base is not object
            )[:MAX_BASE_CLASSES]
            kind = "class"
        elif inspect.isfunction(value) or inspect.isbuiltin(value):
            bases = ()
            kind = "function"
        else:
            return fallback
        return SymbolInfo(
            name=qualname.rsplit(".", 1)[-1],
            qualified=f"{module}.{qualname}",
            module=module,
            kind=kind,
            path=relative,
            line=line,
            signature=_clip(signature, 1_000),
            docstring=docstring,
            bases=bases,
        )

    def _static_symbol(self, name: str) -> SymbolInfo | None:
        qualified = name.strip()
        for symbol in self._public_symbols():
            if symbol.qualified == qualified:
                return symbol
        bare = qualified.rsplit(".", 1)[-1]
        return next(
            (
                symbol
                for symbol in self._public_symbols()
                if symbol.name == bare
            ),
            None,
        )

    def _constructor_details(
        self,
        runtime: Any | None,
        info: SymbolInfo,
    ) -> dict[str, Any] | None:
        if runtime is not None and inspect.isclass(runtime):
            constructor = _class_constructor(runtime)
            if constructor is not None:
                owner, signature = constructor
                owner_name = f"{owner.__module__}.{owner.__qualname__}"
                return {
                    "signature": _clip(str(signature), 1_000),
                    "defined_on": owner_name,
                    "inherited": owner is not runtime,
                    "parameters": _parameter_details(
                        signature,
                        owner_name,
                    ),
                }

        if not _is_uninformative_signature(info.signature):
            return {
                "signature": info.signature,
                "defined_on": info.qualified,
                "inherited": False,
                "parameters": [],
            }

        current = info
        seen: set[str] = set()
        while current.qualified not in seen:
            seen.add(current.qualified)
            base = next(
                (
                    self._static_symbol(base_name)
                    for base_name in current.bases
                    if self._static_symbol(base_name) is not None
                ),
                None,
            )
            if base is None:
                break
            if not _is_uninformative_signature(base.signature):
                return {
                    "signature": base.signature,
                    "defined_on": base.qualified,
                    "inherited": True,
                    "parameters": [],
                }
            current = base
        return None

    def _related_symbols(
        self,
        requested: str,
        info: SymbolInfo,
        runtime: Any | None,
    ) -> list[dict[str, Any]]:
        related: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        def add(symbol: str, relation: str) -> None:
            key = (symbol, relation)
            if (
                not symbol
                or symbol == info.qualified
                or key in seen
                or len(related) >= MAX_RELATED_SYMBOLS
            ):
                return
            seen.add(key)
            related.append(
                {"symbol": _clip(symbol, 320), "relation": relation}
            )

        normalized = _normalise_symbol(requested)
        if normalized != info.qualified:
            add(normalized, "alias")

        for base_name in info.bases:
            base = self._static_symbol(base_name)
            add(base.qualified if base else base_name, "base_class")

        if runtime is not None:
            modules = {
                "deeptrack",
                info.module,
                info.module.rsplit(".", 1)[0],
            }
            for module_name in sorted(modules):
                try:
                    module = importlib.import_module(module_name)
                except Exception:  # pragma: no cover - optional imports vary
                    continue
                for alias, value in vars(module).items():
                    if alias.startswith("_") or value is not runtime:
                        continue
                    add(f"{module_name}.{alias}", "alias")

        for candidate in self._public_symbols():
            if (
                candidate.name.casefold() == info.name.casefold()
                and candidate.qualified != info.qualified
            ):
                add(candidate.qualified, "same_name")

        for module in sorted(self._module_names):
            leaf = module.rsplit(".", 1)[-1]
            if leaf.casefold() == info.name.casefold():
                add(module, "similarly_named_module")

        return related

    def _resolve_symbol(
        self,
        symbol: str,
    ) -> tuple[SymbolInfo | None, Any | None, dict[str, Any] | None]:
        if not isinstance(symbol, str) or not symbol.strip():
            return (
                None,
                None,
                _error("invalid_symbol", "symbol must be a non-empty name"),
            )
        normalized = _normalise_symbol(symbol)
        if len(normalized) > MAX_QUERY_CHARS or not _SYMBOL_PATTERN.fullmatch(
            normalized
        ):
            return (
                None,
                None,
                _error(
                    "invalid_symbol",
                    "symbol must be a dotted Python name inside the "
                    "deeptrack package",
                ),
            )
        if normalized == "deeptrack":
            return (
                None,
                None,
                _error(
                    "symbol_not_found",
                    "The package itself is not a symbol",
                ),
            )

        bare = normalized.removeprefix("deeptrack.")
        final_name = bare.rsplit(".", 1)[-1]
        fallback: SymbolInfo | None = None
        for candidate in self._symbols:
            if candidate.qualified == normalized:
                fallback = candidate
                break
            if bare == candidate.name:
                fallback = candidate
                break
        runtime = self._runtime_object(normalized)
        info = (
            self._runtime_info(runtime, fallback)
            if runtime is not None
            else fallback
        )
        if info is not None:
            return info, runtime, None

        suggestions = difflib.get_close_matches(
            final_name,
            sorted({candidate.name for candidate in self._symbols}),
            n=5,
            cutoff=0.45,
        )
        suggestion_symbols = [
            candidate.qualified
            for candidate in self._symbols
            if candidate.name in suggestions
        ][:5]
        return (
            None,
            None,
            _error(
                "symbol_not_found",
                "Unknown DeepTrack symbol: "
                f"{_clip(str(symbol), MAX_QUERY_CHARS)}",
                suggestions=suggestion_symbols,
            ),
        )

    def inspect_symbol(self, symbol: str) -> dict[str, Any]:
        self._refresh()
        info, runtime, error = self._resolve_symbol(symbol)
        if error is not None or info is None:
            display_symbol = _clip(str(symbol), MAX_QUERY_CHARS)
            return error or _error(
                "symbol_not_found",
                f"Unknown DeepTrack symbol: {display_symbol}",
            )
        related = self.find_examples(info.name, limit=5)
        result: dict[str, Any] = {
            "requested_symbol": symbol,
            "fully_qualified_symbol": info.qualified,
            "signature": info.signature,
            "docstring": _clip(info.docstring, MAX_DOCSTRING_CHARS),
            "module": info.module,
            "source_file": info.path,
            "source_line": info.line,
            "base_classes": list(info.bases),
            "related_examples": related.get("results", []),
            "related_symbols": self._related_symbols(
                symbol,
                info,
                runtime,
            ),
        }
        constructor = self._constructor_details(runtime, info)
        if constructor is not None:
            result["constructor"] = constructor
            if constructor["inherited"]:
                result["inherited_parameters"] = constructor["parameters"]
        return result

    def _canonical_category(self, category: str | None) -> str | None:
        if category is None:
            return None
        if not isinstance(category, str):
            return None
        key = " ".join(category.casefold().replace("_", " ").split())
        if key in _CATEGORY_MODULES:
            return key
        return _CATEGORY_ALIASES.get(key)

    def _runtime_module_symbols(self, module_name: str) -> list[SymbolInfo]:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - optional package failures vary
            return []
        names = getattr(module, "__all__", ())
        if not isinstance(names, (list, tuple)):
            names = [name for name in dir(module) if not name.startswith("_")]
        found: list[SymbolInfo] = []
        for name in names:
            if not isinstance(name, str) or name.startswith("_"):
                continue
            try:
                value = getattr(module, name)
            except AttributeError:
                continue
            if not (inspect.isclass(value) or inspect.isfunction(value)):
                continue
            if not getattr(value, "__module__", "").startswith("deeptrack"):
                continue
            info = self._runtime_info(value, None)
            if info is not None:
                found.append(info)
        return found

    def list_components(self, category: str | None = None) -> dict[str, Any]:
        self._refresh()
        canonical = self._canonical_category(category)
        if category is not None and canonical is None:
            return _error(
                "unknown_category",
                "Unknown component category: "
                f"{_clip(str(category), MAX_QUERY_CHARS)}",
                available_categories=list(_CATEGORY_MODULES),
            )

        if canonical is None:
            prefixes: tuple[str, ...] = tuple(
                prefix
                for values in _CATEGORY_MODULES.values()
                for prefix in values
            )
        else:
            prefixes = _CATEGORY_MODULES[canonical]

        infos = [
            info
            for info in self._symbols
            if any(
                info.module == prefix or info.module.startswith(prefix + ".")
                for prefix in prefixes
            )
        ]
        for prefix in prefixes:
            infos.extend(self._runtime_module_symbols(prefix))
        unique: dict[str, SymbolInfo] = {
            info.qualified: info for info in infos
        }
        infos = sorted(
            unique.values(),
            key=lambda info: info.qualified.casefold(),
        )

        components = []
        for info in infos[:MAX_COMPONENTS]:
            components.append(
                {
                    "name": info.name,
                    "qualified_symbol": info.qualified,
                    "module": info.module,
                    "kind": info.kind,
                    "summary": _summary(info.docstring),
                    "source_file": info.path,
                }
            )
        response: dict[str, Any] = {
            "category": canonical,
            "available_categories": list(_CATEGORY_MODULES),
            "components": components,
            "component_count": len(infos),
            "truncated": len(infos) > MAX_COMPONENTS,
        }
        if canonical is None:
            response["category_counts"] = {
                name: sum(
                    1
                    for info in infos
                    if any(
                        info.module == prefix
                        or info.module.startswith(prefix + ".")
                        for prefix in prefixes_for_category
                    )
                )
                for name, prefixes_for_category in _CATEGORY_MODULES.items()
            }
        return response

    def get_source(self, symbol: str, max_lines: int = 120) -> dict[str, Any]:
        self._refresh()
        info, _, error = self._resolve_symbol(symbol)
        if error is not None or info is None:
            return error or _error(
                "symbol_not_found",
                "Unknown DeepTrack symbol: "
                f"{_clip(str(symbol), MAX_QUERY_CHARS)}",
            )
        path = self._safe_file(info.path)
        if path is None:
            return _error(
                "source_unavailable",
                "The resolved source file is outside this repository or "
                "unavailable",
            )
        try:
            requested_lines = int(max_lines)
        except (TypeError, ValueError):
            requested_lines = 120
        effective_lines = max(1, min(requested_lines, MAX_SOURCE_LINES))
        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError as file_error:
            return _error("source_unavailable", str(file_error))
        start = max(0, (info.line or 1) - 1)
        end = min(len(lines), start + effective_lines)
        source = _clip("\n".join(lines[start:end]), MAX_SOURCE_CHARS)
        return {
            "fully_qualified_symbol": info.qualified,
            "source_file": info.path,
            "start_line": start + 1,
            "end_line": end,
            "requested_max_lines": requested_lines,
            "max_lines": effective_lines,
            "truncated": end < len(lines) or len(source) >= MAX_SOURCE_CHARS,
            "source": source,
        }


def _require_mcp() -> Any:
    if MCPServer is None:
        raise RuntimeError(
            "The optional MCP dependency is not installed. Install it with "
            '`python -m pip install "deeptrack[mcp]"`.'
        )
    return MCPServer


def create_server(repository_root: str | Path | None = None) -> Any:
    """Create a read-only MCPServer bound to a DeepTrack2 checkout."""

    server_type = _require_mcp()
    knowledge = RepositoryKnowledge(repository_root)
    server = server_type(
        "deeptrack-repository",
        description=(
            "Read-only factual discovery for the current DeepTrack2 checkout."
        ),
        instructions=(
            "Use these tools to inspect the checked-out DeepTrack2 API, "
            "source, tests, documentation, and tutorial examples. No tool "
            "modifies files or executes notebook/user code."
        ),
        version="",
    )

    @server.tool(structured_output=True)
    def search_deeptrack(query: str, limit: int = 8) -> dict[str, Any]:
        """Search DeepTrack2 APIs, docstrings, docs, tutorials, and tests."""

        return knowledge.search(query, limit)

    @server.tool(structured_output=True)
    def inspect_symbol(symbol: str) -> dict[str, Any]:
        """Inspect a public symbol using runtime or static metadata."""

        return knowledge.inspect_symbol(symbol)

    @server.tool(structured_output=True)
    def find_examples(query: str, limit: int = 5) -> dict[str, Any]:
        """Find relevant cells in DeepTrack2 tutorial notebooks."""

        return knowledge.find_examples(query, limit)

    @server.tool(structured_output=True)
    def list_components(category: str | None = None) -> dict[str, Any]:
        """List dynamically discovered DeepTrack2 component families."""

        return knowledge.list_components(category)

    @server.tool(structured_output=True)
    def get_source(symbol: str, max_lines: int = 120) -> dict[str, Any]:
        """Return a bounded source excerpt for a DeepTrack2 symbol."""

        return knowledge.get_source(symbol, max_lines)

    return server


if MCPServer is not None:
    try:
        # Expose the conventional variable for `mcp dev` while doing no index
        # work until a client calls a tool.
        mcp = create_server()
    except RuntimeError:
        # An installed wheel may not contain a repository checkout.  `main`
        # retries and emits a useful stderr error when launched in that state.
        mcp = None
else:
    mcp = None


def main() -> None:
    """Run the server over stdio, the default MCP transport."""

    try:
        server = create_server()
    except Exception as error:  # pragma: no cover - process-level error path
        print(f"deeptrack-mcp: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    server.run("stdio")


__all__ = [
    "RepositoryKnowledge",
    "create_server",
    "main",
    "resolve_repository_root",
]


if __name__ == "__main__":  # pragma: no cover - covered through stdio client
    main()

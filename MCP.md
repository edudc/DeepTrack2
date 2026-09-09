# DeepTrack2 repository MCP server

DeepTrack2 includes an optional, read-only Model Context Protocol server for
coding agents. It discovers facts from the current checkout at request time:
public Python symbols and docstrings, source, documentation, tests, and
tutorial notebook markdown/code cells. It does not execute notebooks or
caller-supplied Python, modify files, install packages, use the network, or
run shell commands.

## Installation and start-up

Install the optional dependency and the console entry point from a checkout:

```bash
python -m pip install -e ".[mcp]"
deeptrack-mcp
```

The server uses stdio by default. The equivalent module invocation is:

```bash
python -m deeptrack.mcp_server
```

The repository root is resolved from the installed module location before the
working directory, so starting it from another directory works with an
editable install. Set `DEEPTRACK_REPOSITORY` if the checkout is in an unusual
layout.

## Tools

- `search_deeptrack(query, limit=8)` searches API names, Python docstrings,
  Markdown/reStructuredText documentation, notebook markdown and code cells,
  and tests. Results include a kind, title, repository path, bounded excerpt,
  and lexical relevance details.
- `inspect_symbol(symbol)` accepts names such as
  `deeptrack.MieSphere`, `deeptrack.Brightfield`, `deeptrack.ISCAT`, and
  `deeptrack.Holography`. It returns the fully qualified symbol, signature,
  bounded docstring, module, source file, base classes, constructor details
  (including inherited parameters when needed), related symbols, and related
  tutorial cells.
- `find_examples(query, limit=5)` returns only relevant tutorial notebook
  cells, with notebook paths and cell numbers. Public symbol queries such as
  `Holography` prefer calls to that public class, while an explicit module
  query such as `deeptrack.optical.holography` targets the lower-level module.
  Code cells may include an immediately preceding explanatory markdown cell.
- `list_components(category=None)` discovers public component families from
  package modules. Supported family filters include `scatterers`, `optics`,
  `aberrations`, `noise/features`, `sequences`, and `pytorch` (with common
  aliases such as `features` and `pytorch integration`).
- `get_source(symbol, max_lines=120)` returns a bounded source excerpt. Source
  paths are validated to remain inside the repository.

Example calls, expressed as MCP tool arguments:

```json
{"query": "MieSphere brightfield", "limit": 3}
{"symbol": "deeptrack.Brightfield"}
{"query": "holography", "limit": 5}
{"category": "scatterers"}
{"symbol": "deeptrack.MieSphere", "max_lines": 40}
```

## Codex configuration

The Codex CLI can register a local stdio server with:

```bash
codex mcp add deeptrack -- deeptrack-mcp
```

Or add a project-scoped `.codex/config.toml` entry (use an absolute command
path when the environment is not on `PATH`):

```toml
[mcp_servers.deeptrack]
command = "/absolute/path/to/deeptrack-mcp"
cwd = "/absolute/path/to/DeepTrack2"
startup_timeout_sec = 10
tool_timeout_sec = 60
```

Run `codex mcp list` or `/mcp` in the Codex TUI to verify the connection. The
official Codex MCP guide documents the CLI and `config.toml` forms:
<https://developers.openai.com/codex/mcp>.

## Development and Inspector

Install the optional extra, then open the server in the MCP Inspector:

```bash
python -m pip install -e ".[mcp]"
uv run --with "mcp[cli]" mcp dev --with "mcp[cli]" deeptrack/mcp_server.py
```

The first `--with` installs the CLI that opens the Inspector. The second is
passed through by `mcp dev` to the child stdio process; both are needed when
using a temporary `uv` environment. If `uv` warns that an active virtual
environment differs from `.venv`, use `uv run --active` when that active
environment is the one you intend to use.

The SDK's in-memory client is used by `tests/test_mcp_server.py` for protocol
tests. Run the focused tests with:

```bash
.venv312/bin/python -m pytest tests/test_mcp_server.py -q
```

The Inspector and SDK v2 documentation are available from the official Python
SDK repository: <https://github.com/modelcontextprotocol/python-sdk>.

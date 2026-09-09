"""Tests for the optional, read-only DeepTrack2 MCP server."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp", reason="MCP SDK v2 is an optional dependency")

from mcp import Client  # noqa: E402
from mcp.client.stdio import (  # noqa: E402
    StdioServerParameters,
    stdio_client,
)

from deeptrack.mcp_server import (  # noqa: E402
    MAX_EXCERPT_CHARS,
    MAX_RESULTS,
    MAX_SOURCE_LINES,
    RepositoryKnowledge,
    create_server,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {
    "search_deeptrack",
    "inspect_symbol",
    "find_examples",
    "list_components",
    "get_source",
}


def _structured(result):
    payload = getattr(result, "structured_content", None)
    assert isinstance(payload, dict), result
    return payload


def test_tools_list_exposes_expected_tools() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == EXPECTED_TOOLS

    asyncio.run(run())


def test_search_finds_miesphere_and_brightfield_content() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            result = await client.call_tool(
                "search_deeptrack",
                {"query": "MieSphere brightfield"},
            )
            payload = _structured(result)
            assert payload["results"]
            searchable = " ".join(
                f"{item['title']} {item['path']} {item['excerpt']}"
                for item in payload["results"]
            ).casefold()
            assert "miesphere" in searchable
            assert "brightfield" in searchable

    asyncio.run(run())


def test_search_prefers_public_content_over_tests() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            payload = _structured(
                await client.call_tool(
                    "search_deeptrack",
                    {"query": "MieSphere brightfield", "limit": 5},
                )
            )
            first = payload["results"][0]
            assert first["kind"] in {
                "api_symbol",
                "tutorial_code",
                "tutorial_markdown",
            }
            assert not first["path"].startswith("tests/")

    asyncio.run(run())


def test_search_boosts_exact_iscat_symbol() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            payload = _structured(
                await client.call_tool(
                    "search_deeptrack",
                    {"query": "ISCAT illumination angle", "limit": 5},
                )
            )
            first = payload["results"][0]
            assert first["kind"] == "api_symbol"
            assert first["title"].endswith(".ISCAT")
            assert first["relevance"]["coverage"] == 1.0

    asyncio.run(run())


def test_exact_symbol_queries_are_boosted() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            payload = _structured(
                await client.call_tool(
                    "search_deeptrack",
                    {"query": "MieSphere", "limit": 1},
                )
            )
            first = payload["results"][0]
            assert first["kind"] == "api_symbol"
            assert first["title"].endswith(".MieSphere")
            assert first["relevance"]["exact_symbol_query"] is True

    asyncio.run(run())


@pytest.mark.parametrize(
    "symbol",
    ["deeptrack.MieSphere", "deeptrack.Brightfield"],
)
def test_inspect_symbol_works(symbol: str) -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            payload = _structured(
                await client.call_tool("inspect_symbol", {"symbol": symbol})
            )
            assert payload["fully_qualified_symbol"].endswith(
                symbol.split(".")[-1]
            )
            assert payload["signature"].startswith("(")
            assert payload["docstring"]
            assert payload["module"].startswith("deeptrack.")
            assert payload["source_file"].startswith("deeptrack/")

    asyncio.run(run())


def test_find_examples_finds_holography() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            payload = _structured(
                await client.call_tool(
                    "find_examples",
                    {"query": "holography", "limit": 5},
                )
            )
            assert payload["results"]
            assert any(
                "tutorials/" in item["notebook"] for item in payload["results"]
            )
            assert all("cell-" in item["path"] for item in payload["results"])

    asyncio.run(run())


def test_find_examples_exposes_iscat_code_and_context() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            payload = _structured(
                await client.call_tool(
                    "find_examples",
                    {"query": "ISCAT", "limit": 5},
                )
            )
            first = payload["results"][0]
            assert first["kind"] == "tutorial_code"
            assert "dt.ISCAT(" in first["excerpt"]
            assert first["related_cells"][0]["role"] == (
                "preceding_explanation"
            )

    asyncio.run(run())


def test_find_examples_distinguishes_public_holography_and_module() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            public = _structured(
                await client.call_tool(
                    "find_examples",
                    {"query": "Holography", "limit": 5},
                )
            )
            module = _structured(
                await client.call_tool(
                    "find_examples",
                    {"query": "deeptrack.optical.holography", "limit": 5},
                )
            )
            bare_module = _structured(
                await client.call_tool(
                    "find_examples",
                    {"query": "holography", "limit": 5},
                )
            )
            assert public["target_mode"] == "public_symbol"
            assert public["results"][0]["kind"] == "tutorial_code"
            assert "dt.Holography(" in public["results"][0]["excerpt"]
            assert module["target_mode"] == "module"
            assert module["results"][0]["kind"] == "tutorial_code"
            assert (
                "holography.FourierTransform"
                in module["results"][0]["excerpt"]
            )
            assert bare_module["target_mode"] == "module"
            assert (
                "holography.FourierTransform"
                in bare_module["results"][0]["excerpt"]
            )

    asyncio.run(run())


def test_find_examples_diversifies_notebooks() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            payload = _structured(
                await client.call_tool(
                    "find_examples",
                    {"query": "MieSphere", "limit": 5},
                )
            )
            notebooks = {item["notebook"] for item in payload["results"]}
            assert len(notebooks) >= 3

    asyncio.run(run())


def test_unknown_symbol_returns_useful_error() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            payload = _structured(
                await client.call_tool(
                    "inspect_symbol",
                    {"symbol": "deeptrack.DoesNotExist"},
                )
            )
            assert payload["error"]["code"] == "symbol_not_found"
            assert "Unknown DeepTrack symbol" in payload["error"]["message"]
            assert "suggestions" in payload["error"]

    asyncio.run(run())


def test_inspect_symbol_reports_inheritance_and_related_symbols() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            payload = _structured(
                await client.call_tool(
                    "inspect_symbol",
                    {"symbol": "deeptrack.Holography"},
                )
            )
            assert payload["constructor"]["inherited"] is True
            assert any(
                item["name"] == "NA"
                for item in payload["inherited_parameters"]
            )
            relations = {
                item["relation"] for item in payload["related_symbols"]
            }
            related_names = {
                item["symbol"] for item in payload["related_symbols"]
            }
            assert "base_class" in relations
            assert "alias" in relations
            assert "deeptrack.optical.holography" in related_names

    asyncio.run(run())


def test_source_access_cannot_escape_repository() -> None:
    knowledge = RepositoryKnowledge(ROOT)
    payload = knowledge.get_source("../../../etc/passwd")
    assert payload["error"]["code"] == "invalid_symbol"


def test_outputs_are_bounded() -> None:
    async def run() -> None:
        async with Client(create_server(ROOT)) as client:
            search = _structured(
                await client.call_tool(
                    "search_deeptrack",
                    {"query": "deeptrack", "limit": 10_000},
                )
            )
            assert len(search["results"]) <= MAX_RESULTS
            assert all(
                len(item["excerpt"]) <= MAX_EXCERPT_CHARS
                for item in search["results"]
            )

            source = _structured(
                await client.call_tool(
                    "get_source",
                    {"symbol": "deeptrack.MieSphere", "max_lines": 10_000},
                )
            )
            assert source["max_lines"] <= MAX_SOURCE_LINES
            assert len(source["source"].splitlines()) <= MAX_SOURCE_LINES

    asyncio.run(run())


def test_server_starts_over_stdio() -> None:
    async def run() -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "deeptrack.mcp_server"],
            cwd=ROOT.parent,
            env=environment,
        )
        async with Client(stdio_client(parameters)) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == EXPECTED_TOOLS

    asyncio.run(run())


def test_module_loads_with_mcp_cli_direct_loader() -> None:
    """Match the file-loading path used by ``mcp dev``."""

    module_name = "_deeptrack_mcp_direct_loader"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "deeptrack" / "mcp_server.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    # The MCP CLI executes the module without first adding it to
    # sys.modules.  Keep this regression test aligned with that behavior.
    spec.loader.exec_module(module)

    assert module.mcp is not None

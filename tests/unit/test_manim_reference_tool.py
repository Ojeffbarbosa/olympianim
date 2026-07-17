"""Tests for the native LangChain Manim documentation tool."""

from __future__ import annotations

import json

import httpx
import pytest

from olympianim.tools.manim_reference import (
    MANIM_DOCS_BASE_URL,
    ManimDocsSearch,
    _ensure_official_url,
)


def _client_with_official_docs() -> tuple[httpx.Client, list[str]]:
    calls: list[str] = []
    index = {
        "docnames": ["reference/manim.Scene", "reference/manim.mobject.geometry.Circle"],
        "filenames": ["reference/manim.Scene.rst", "reference/Circle.rst"],
        "titles": ["Scene", "Circle"],
        "terms": {"scene": 0, "play": 0, "circle": 1},
        "objects": {"manim.scene.scene.Scene": [[0, 2, 1, "", "play"]]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("searchindex.js"):
            return httpx.Response(
                200,
                text=f"Search.setIndex({json.dumps(index)})",
                request=request,
            )
        return httpx.Response(
            200,
            text=(
                "<html><main><h1>Scene</h1>"
                "<p>Scene.play plays animations and updates mobjects in the scene.</p>"
                "</main></html>"
            ),
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_search_uses_official_html_page_and_ranks_exact_api() -> None:
    client, calls = _client_with_official_docs()

    results = ManimDocsSearch(client).search("Scene.play", limit=1)

    assert len(results) == 1
    assert results[0].title == "Scene"
    assert results[0].url == f"{MANIM_DOCS_BASE_URL}reference/manim.Scene.html"
    assert "Scene.play" in results[0].excerpt
    assert all(url.startswith(MANIM_DOCS_BASE_URL) for url in calls)


def test_search_caches_online_index_in_memory() -> None:
    client, calls = _client_with_official_docs()
    search = ManimDocsSearch(client)

    search.search("Scene", limit=1)
    search.search("Circle", limit=1)

    assert sum(url.endswith("searchindex.js") for url in calls) == 1


def test_search_rejects_non_official_or_non_stable_urls() -> None:
    with pytest.raises(ValueError, match="documentação oficial"):
        _ensure_official_url("https://example.com/en/stable/Scene.html")
    with pytest.raises(ValueError, match="documentação oficial"):
        _ensure_official_url("https://docs.manim.community/en/latest/Scene.html")

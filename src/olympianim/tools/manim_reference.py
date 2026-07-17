"""Online search over the official Manim Community documentation."""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from langchain.tools import tool
from pydantic import BaseModel, Field

MANIM_DOCS_BASE_URL = "https://docs.manim.community/en/stable/"
MANIM_SEARCH_INDEX_URL = urljoin(MANIM_DOCS_BASE_URL, "searchindex.js")
_ALLOWED_HOST = "docs.manim.community"
_INDEX_PREFIX = "Search.setIndex("
_CACHE_TTL_SECONDS = 20 * 60
_REQUEST_TIMEOUT_SECONDS = 8.0
_MAX_RESULTS = 5


class ManimReferenceInput(BaseModel):
    """Input accepted by the official Manim reference search."""

    query: str = Field(
        min_length=2,
        max_length=160,
        description="Manim class, method, error, or API concept to look up.",
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=_MAX_RESULTS,
        description="Maximum number of official documentation results.",
    )


@dataclass(frozen=True)
class ManimReferenceResult:
    """One page selected from the official documentation index."""

    title: str
    url: str
    excerpt: str
    score: int


class ManimDocsSearch:
    """Search the Sphinx index published by the official Manim website."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client
        self._cached_index: Mapping[str, Any] | None = None
        self._cached_at = 0.0
        self._documentation_version = ""
        self._lock = threading.Lock()

    def search(self, query: str, *, limit: int = 3) -> list[ManimReferenceResult]:
        """Return ranked official pages with short relevant excerpts."""
        index = self._get_index()
        tokens = _tokens(query)
        if not tokens:
            return []

        docnames = _string_list(index.get("docnames"))
        titles = _string_list(index.get("titles"))
        scores: defaultdict[int, int] = defaultdict(int)

        normalized_query = _normalize(query)
        for doc_id, title in enumerate(titles):
            normalized_title = _normalize(title)
            normalized_docname = _normalize(docnames[doc_id]) if doc_id < len(docnames) else ""
            if normalized_query in normalized_title:
                scores[doc_id] += 30
            for token in tokens:
                if token in normalized_title:
                    scores[doc_id] += 12
                if token in normalized_docname:
                    scores[doc_id] += 6

        for term, references in _mapping(index.get("terms")).items():
            normalized_term = _normalize(str(term))
            matching_tokens = [token for token in tokens if token in normalized_term]
            if not matching_tokens:
                continue
            weight = 8 + 2 * len(matching_tokens)
            for doc_id in _document_ids(references):
                scores[doc_id] += weight

        for prefix, entries in _mapping(index.get("objects")).items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, list) or len(entry) < 5 or not isinstance(entry[0], int):
                    continue
                name = str(entry[4])
                qualified_name = _normalize(f"{prefix}.{name}")
                if qualified_name.endswith(normalized_query):
                    scores[entry[0]] += 100
                elif all(token in qualified_name for token in tokens):
                    scores[entry[0]] += 50
                elif tokens[-1] == _normalize(name):
                    scores[entry[0]] += 25

        ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        results: list[ManimReferenceResult] = []
        for doc_id in ranked:
            if len(results) >= min(limit, _MAX_RESULTS):
                break
            if doc_id >= len(docnames):
                continue
            url = _official_url(f"{docnames[doc_id]}.html")
            excerpt = self._fetch_excerpt(url, tokens)
            results.append(
                ManimReferenceResult(
                    title=titles[doc_id] if doc_id < len(titles) else docnames[doc_id],
                    url=url,
                    excerpt=excerpt,
                    score=scores[doc_id],
                )
            )
        return results

    def _get_index(self) -> Mapping[str, Any]:
        now = time.monotonic()
        with self._lock:
            if self._cached_index is not None and now - self._cached_at < _CACHE_TTL_SECONDS:
                return self._cached_index
            response = self._get(MANIM_SEARCH_INDEX_URL)
            index = _parse_search_index(response.text)
            self._cached_index = index
            self._cached_at = now
            return index

    def _fetch_excerpt(self, url: str, tokens: tuple[str, ...]) -> str:
        response = self._get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        version_match = re.search(r"Manim Community v([0-9][0-9.]+)", title)
        if version_match is not None:
            self._documentation_version = version_match.group(1)
        container = soup.select_one("main") or soup.select_one("div[role='main']") or soup
        candidates: list[tuple[int, str]] = []
        for element in container.select("h1, h2, h3, p, dt, dd"):
            text = " ".join(element.get_text(" ", strip=True).split())
            if len(text) < 25:
                continue
            normalized = _normalize(text)
            score = sum(1 for token in tokens if token in normalized)
            candidates.append((score, text))
        if not candidates:
            return "Página oficial encontrada; sem trecho textual disponível."
        _, excerpt = max(candidates, key=lambda item: (item[0], len(item[1])))
        return excerpt[:600]

    def _get(self, url: str) -> httpx.Response:
        _ensure_official_url(url)
        if self._client is not None:
            response = self._client.get(url)
        else:
            response = httpx.get(
                url,
                timeout=_REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
                headers={"User-Agent": "Olympianim/0.1 ManimReferenceTool"},
            )
        response.raise_for_status()
        _ensure_official_url(str(response.url))
        return response


_SEARCH = ManimDocsSearch()


@tool(
    "search_manim_reference",
    args_schema=ManimReferenceInput,
    response_format="content_and_artifact",
)
def search_manim_reference(query: str, limit: int = 3) -> tuple[str, dict[str, Any]]:
    """Search only the current official Manim Community documentation.

    Use this tool to confirm Manim classes, methods, signatures, layout APIs,
    animation behavior, or the documented meaning of a render error. Search
    before guessing an unfamiliar API. It never searches tutorials or local
    examples.
    """
    results = _SEARCH.search(query, limit=limit)
    artifact: dict[str, Any] = {
        "query": query,
        "source": MANIM_DOCS_BASE_URL,
        "results": [
            {"title": item.title, "url": item.url, "excerpt": item.excerpt} for item in results
        ],
    }
    try:
        installed_version = version("manim")
    except PackageNotFoundError:
        installed_version = "não instalado"
    documentation_version = _SEARCH._documentation_version
    artifact["installed_version"] = installed_version
    artifact["documentation_version"] = documentation_version or "stable"
    artifact["version_match"] = bool(
        documentation_version and installed_version == documentation_version
    )
    if not results:
        return "Nenhum resultado encontrado na documentação oficial do Manim.", artifact
    content = "\n\n".join(
        f"[{number}] {item.title}\nURL: {item.url}\nTrecho: {item.excerpt}"
        for number, item in enumerate(results, start=1)
    )
    if documentation_version and installed_version != documentation_version:
        content = (
            "Aviso de versão: Manim instalado "
            f"{installed_version}; documentação stable {documentation_version}.\n\n{content}"
        )
    return content, artifact


def _parse_search_index(source: str) -> Mapping[str, Any]:
    stripped = source.strip()
    if not stripped.startswith(_INDEX_PREFIX) or not stripped.endswith(")"):
        raise ValueError("Índice de busca da documentação do Manim em formato inválido.")
    value = json.loads(stripped[len(_INDEX_PREFIX) : -1])
    if not isinstance(value, Mapping):
        raise ValueError("Índice de busca da documentação do Manim sem objeto raiz.")
    return value


def _official_url(filename: str) -> str:
    url = urljoin(MANIM_DOCS_BASE_URL, filename)
    _ensure_official_url(url)
    return url


def _ensure_official_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _ALLOWED_HOST
        or not parsed.path.startswith("/en/stable/")
    ):
        raise ValueError("A busca Manim recusou uma URL fora da documentação oficial stable.")


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"[a-z0-9_]+", _normalize(value))))


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return normalized.casefold()


def _document_ids(value: Any) -> set[int]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        result: set[int] = set()
        for item in value:
            result.update(_document_ids(item))
        return result
    return set()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]

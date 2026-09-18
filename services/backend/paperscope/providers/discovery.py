from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import ClassVar

import httpx

from paperscope.config import settings
from paperscope.providers.contracts import ProviderConfigurationError, RetryableProviderError
from paperscope.providers.nim import NimProvider
from paperscope.schemas import PaperCandidate, ResearchIntent


@dataclass(frozen=True)
class DiscoveredPaper:
    id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str
    source: str
    doi: str | None = None
    arxiv_id: str | None = None
    source_url: str | None = None


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _stable_candidate_id(title: str, external_id: str | None = None) -> str:
    seed = external_id or _title_key(title)
    return "paper-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _normalized_identifier(value: str) -> str:
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", value.strip().lower()).rstrip(".")


def _title_similarity(left: str, right: str) -> float:
    left_key = _title_key(left)
    right_key = _title_key(right)
    if not left_key or not right_key:
        return 0.0
    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return max(overlap, SequenceMatcher(None, left_key, right_key).ratio())


def _same_paper(left: DiscoveredPaper, right: DiscoveredPaper) -> bool:
    if left.doi and right.doi and _normalized_identifier(left.doi) == _normalized_identifier(right.doi):
        return True
    if left.arxiv_id and right.arxiv_id:
        left_id = re.sub(r"v\d+$", "", _normalized_identifier(left.arxiv_id))
        right_id = re.sub(r"v\d+$", "", _normalized_identifier(right.arxiv_id))
        if left_id == right_id:
            return True
    return _title_similarity(left.title, right.title) >= 0.92


class SemanticScholarClient:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    async def search(self, query: str, limit: int = 20) -> list[DiscoveredPaper]:
        headers = {"x-api-key": settings.semantic_scholar_api_key} if settings.semantic_scholar_api_key else {}
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,authors,year,venue,abstract,externalIds,openAccessPdf",
        }
        try:
            async with httpx.AsyncClient(timeout=settings.discovery_http_timeout_seconds) as client:
                response = await client.get(self.url, params=params, headers=headers)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RetryableProviderError(f"Semantic Scholar returned {response.status_code}")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RetryableProviderError(f"Semantic Scholar request failed: {exc}") from exc

        results: list[DiscoveredPaper] = []
        for item in response.json().get("data", []):
            external_ids = item.get("externalIds") or {}
            arxiv_id = external_ids.get("ArXiv")
            open_access_pdf = item.get("openAccessPdf") or {}
            results.append(
                DiscoveredPaper(
                    id=_stable_candidate_id(item.get("title", ""), item.get("paperId")),
                    title=item.get("title") or "Untitled paper",
                    authors=[author.get("name", "Unknown author") for author in item.get("authors", [])],
                    year=item.get("year"),
                    venue=item.get("venue"),
                    abstract=item.get("abstract") or "No abstract was supplied.",
                    source="Semantic Scholar",
                    doi=external_ids.get("DOI"),
                    arxiv_id=arxiv_id,
                    source_url=open_access_pdf.get("url") or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None),
                )
            )
        return results


class ArxivClient:
    url = "https://export.arxiv.org/api/query"
    namespace: ClassVar[dict[str, str]] = {"atom": "http://www.w3.org/2005/Atom"}

    async def search(self, query: str, limit: int = 20) -> list[DiscoveredPaper]:
        try:
            async with httpx.AsyncClient(timeout=settings.discovery_http_timeout_seconds) as client:
                response = await client.get(self.url, params={"search_query": f"all:{query}", "max_results": limit})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RetryableProviderError(f"arXiv request failed: {exc}") from exc

        root = ET.fromstring(response.text)
        papers: list[DiscoveredPaper] = []
        for entry in root.findall("atom:entry", self.namespace):
            raw_id = entry.findtext("atom:id", default="", namespaces=self.namespace)
            arxiv_id = raw_id.rsplit("/", 1)[-1]
            title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=self.namespace)).strip()
            published = entry.findtext("atom:published", default="", namespaces=self.namespace)
            year = int(published[:4]) if published[:4].isdigit() else None
            pdf_url = next(
                (
                    link.get("href")
                    for link in entry.findall("atom:link", self.namespace)
                    if link.get("title") == "pdf" or link.get("type") == "application/pdf"
                ),
                None,
            )
            source_url = pdf_url or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None)
            papers.append(
                DiscoveredPaper(
                    id=_stable_candidate_id(title, arxiv_id),
                    title=title,
                    authors=[author.findtext("atom:name", default="Unknown author", namespaces=self.namespace) for author in entry.findall("atom:author", self.namespace)],
                    year=year,
                    venue="arXiv",
                    abstract=re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=self.namespace)).strip(),
                    source="arXiv",
                    arxiv_id=arxiv_id,
                    source_url=source_url,
                )
            )
        return papers


def heuristic_intent(query: str) -> ResearchIntent:
    clean = re.sub(r"\s+", " ", query.strip())
    lower = clean.lower()
    expansions: list[str] = []
    replacements = {
        "moe": "mixture of experts",
        "rag": "retrieval augmented generation",
        "vlm": "vision language model",
        "llm": "large language model",
    }
    expanded = lower
    for short, long in replacements.items():
        pattern = re.compile(rf"\b{re.escape(short)}\b", re.IGNORECASE)
        if pattern.search(lower) and long not in lower:
            expanded = pattern.sub(long, expanded)
    expansions.append(clean)
    if expanded != lower:
        expansions.append(expanded)
    expansions.append(f"{clean} methods and systems")
    expansions.append(f"{clean} benchmark and evaluation")
    words = [word for word in re.findall(r"[a-zA-Z0-9-]+", clean) if len(word) > 2]
    concepts = words[:5]
    return ResearchIntent(
        original_query=query,
        normalized_topic=expanded.title(),
        core_concepts=concepts,
        related_terms=[f"{word} inference" for word in words[:3]],
        methods=["system design", "empirical evaluation"],
        target_problems=["efficiency", "quality", "latency"],
        search_queries=list(dict.fromkeys(expansions))[:5],
    )


class ResearchDiscovery:
    def __init__(self) -> None:
        self.semantic_scholar = SemanticScholarClient()
        self.arxiv = ArxivClient()

    async def interpret(self, query: str, timeout_seconds: float | None = None) -> ResearchIntent:
        intent = heuristic_intent(query)
        if not settings.nvidia_api_key:
            return intent
        timeout = min(
            float(settings.discovery_provider_timeout_seconds),
            timeout_seconds if timeout_seconds is not None else float(settings.discovery_provider_timeout_seconds),
        )
        if timeout <= 0:
            return intent
        prompt = f"""
Interpret this research query and return JSON with exactly these fields:
original_query, normalized_topic, core_concepts, related_terms, methods,
target_problems, search_queries. Use three to five concise search_queries.

        Query: {query}
        """.strip()
        try:
            raw = await asyncio.wait_for(
                NimProvider(
                    settings.rag_primary_model,
                    timeout_seconds=settings.discovery_provider_timeout_seconds,
                ).complete(prompt),
                timeout=timeout,
            )
            return ResearchIntent.model_validate_json(raw)
        except (ProviderConfigurationError, RetryableProviderError, TimeoutError, ValueError):
            return intent

    async def search(self, query: str, limit: int = 10) -> tuple[ResearchIntent, list[PaperCandidate], bool]:
        deadline = time.monotonic() + max(float(settings.discovery_timeout_seconds) - 0.25, 0.5)
        intent = await self.interpret(query, max(deadline - time.monotonic(), 0.0))
        searches: list[list[DiscoveredPaper] | BaseException] = []
        remaining = deadline - time.monotonic()
        if remaining > 0.1:
            try:
                searches = await asyncio.wait_for(
                    asyncio.gather(
                        *(self.semantic_scholar.search(search_query, limit=20) for search_query in intent.search_queries),
                        return_exceptions=True,
                    ),
                    timeout=remaining,
                )
            except TimeoutError as exc:
                searches = [exc]
        papers: list[DiscoveredPaper] = []
        provider_failures: list[BaseException] = []
        successful_search = False
        for result in searches:
            if isinstance(result, list):
                papers.extend(result)
                successful_search = True
            elif isinstance(result, BaseException):
                provider_failures.append(result)
        remaining = deadline - time.monotonic()
        if len(papers) < limit and remaining > 0.1:
            try:
                fallback = await asyncio.wait_for(
                    asyncio.gather(
                        *(self.arxiv.search(search_query, limit=10) for search_query in intent.search_queries[:2]),
                        return_exceptions=True,
                    ),
                    timeout=remaining,
                )
            except TimeoutError as exc:
                fallback = [exc]
            for result in fallback:
                if isinstance(result, list):
                    papers.extend(result)
                    successful_search = True
                elif isinstance(result, BaseException):
                    provider_failures.append(result)

        if not papers and provider_failures and not successful_search:
            raise RetryableProviderError("Research discovery providers are unavailable")

        deduped: dict[str, DiscoveredPaper] = {}
        for paper in papers:
            if paper.doi:
                key = f"doi:{_normalized_identifier(paper.doi)}"
            elif paper.arxiv_id:
                arxiv_key = re.sub(r"v\d+$", "", _normalized_identifier(paper.arxiv_id))
                key = f"arxiv:{arxiv_key}"
            else:
                key = f"title:{_title_key(paper.title)}"
            if key and key not in deduped and not any(_same_paper(paper, existing) for existing in deduped.values()):
                deduped[key] = paper

        candidates = [
            PaperCandidate(
                id=paper.id,
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                venue=paper.venue,
                abstract=paper.abstract,
                relevance_score=max(0.05, 1 - index / max(len(deduped), 1)),
                relevance_reason="Matched the expanded research topic.",
                source=paper.source,  # type: ignore[arg-type]
                doi=paper.doi,
                arxiv_id=paper.arxiv_id,
                source_url=paper.source_url,
            )
            for index, paper in enumerate(deduped.values())
        ]
        remaining = deadline - time.monotonic()
        if remaining <= 0.25:
            return intent, candidates[:limit], False
        ranked, reranked = await self._rerank(intent, candidates[:30], timeout_seconds=remaining)
        return intent, ranked[:limit], reranked

    async def _rerank(
        self,
        intent: ResearchIntent,
        candidates: list[PaperCandidate],
        timeout_seconds: float | None = None,
    ) -> tuple[list[PaperCandidate], bool]:
        if not candidates or not settings.nvidia_api_key:
            return candidates, False
        timeout = min(
            float(settings.discovery_provider_timeout_seconds),
            timeout_seconds if timeout_seconds is not None else float(settings.discovery_provider_timeout_seconds),
        )
        if timeout <= 0:
            return candidates, False
        lines = "\n".join(f"{index}: {item.title}\n{item.abstract[:700]}" for index, item in enumerate(candidates))
        prompt = f"""
Rank these papers for the research topic below. Return JSON array items with index,
score between 0 and 1, and a short reason. Score conceptual relevance, not keyword
overlap. Return all items sorted by descending score.

Topic: {intent.normalized_topic}

Candidates:
{lines}
        """.strip()
        try:
            raw = await asyncio.wait_for(
                NimProvider(
                    settings.rag_primary_model,
                    timeout_seconds=settings.discovery_provider_timeout_seconds,
                ).complete(prompt),
                timeout=timeout,
            )
            ranking = raw[raw.find("[") : raw.rfind("]") + 1]
            values = json.loads(ranking)
            by_index = {int(item["index"]): item for item in values}
            ranked = []
            for index, candidate in enumerate(candidates):
                score = by_index.get(index, {}).get("score", candidate.relevance_score)
                reason = by_index.get(index, {}).get("reason", candidate.relevance_reason)
                ranked.append(candidate.model_copy(update={"relevance_score": max(0, min(1, float(score))), "relevance_reason": str(reason)}))
            ranked.sort(key=lambda item: item.relevance_score, reverse=True)
            return ranked, True
        except (ProviderConfigurationError, RetryableProviderError, TimeoutError, ValueError, TypeError, KeyError):
            return candidates, False

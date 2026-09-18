import pytest

from paperscope.providers.discovery import ArxivClient, DiscoveredPaper, _same_paper, heuristic_intent


def test_heuristic_intent_expands_common_research_abbreviations() -> None:
    intent = heuristic_intent("efficient speculative decoding for MoE inference")
    assert intent.original_query.startswith("efficient")
    assert any("mixture of experts" in query for query in intent.search_queries)
    assert 1 <= len(intent.search_queries) <= 5


def test_heuristic_intent_does_not_expand_abbreviations_inside_words() -> None:
    intent = heuristic_intent("storage optimization")

    assert all("retrieval augmented generation" not in query for query in intent.search_queries)
    assert intent.search_queries[0] == "storage optimization"


def test_discovery_deduplicates_versioned_arxiv_records_and_near_titles() -> None:
    first = DiscoveredPaper(
        id="one",
        title="A Practical Guide to Vector Retrieval",
        authors=[],
        year=2025,
        venue="Semantic Scholar",
        abstract="",
        source="Semantic Scholar",
        arxiv_id="2501.12345v1",
    )
    second = DiscoveredPaper(
        id="two",
        title="A Practical Guide to Vector Retrieval: An Updated Study",
        authors=[],
        year=2025,
        venue="arXiv",
        abstract="",
        source="arXiv",
        arxiv_id="2501.12345v2",
    )

    assert _same_paper(first, second)


@pytest.mark.asyncio
async def test_arxiv_records_expose_a_pdf_source_url(monkeypatch) -> None:
    response_text = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2501.12345v2</id>
        <title> A useful paper </title>
        <published>2025-01-01T00:00:00Z</published>
        <summary>Summary</summary>
        <author><name>Author</name></author>
        <link title="pdf" type="application/pdf" href="https://arxiv.org/pdf/2501.12345v2" />
      </entry>
    </feed>
    """

    class FakeResponse:
        text = response_text

        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("paperscope.providers.discovery.httpx.AsyncClient", lambda **kwargs: FakeClient())
    papers = await ArxivClient().search("useful paper", limit=1)

    assert papers[0].source_url == "https://arxiv.org/pdf/2501.12345v2"

import pytest
from pydantic import ValidationError

from paperscope.processing.ingest import _visual_filename, reconcile_graph
from paperscope.schemas import GraphEdge, GraphNode, PaperGraph


def test_reconcile_graph_discards_unresolvable_evidence_and_edges() -> None:
    graph = PaperGraph(
        nodes=[
            GraphNode(id="method", label="Method", type="process", explanation="The proposed pipeline.", evidence_keys=["page-2", "missing"]),
            GraphNode(id="result", label="Result", type="result", explanation="Reported improvement.", evidence_keys=[]),
        ],
        edges=[
            GraphEdge(id="valid", source="method", target="result", relation="produces"),
            GraphEdge(id="invalid", source="method", target="missing", relation="flows_to"),
        ],
    )

    nodes, edges = reconcile_graph(graph, {"page-2": "page-db-id"})

    assert nodes[0].evidence_keys == ["page-db-id"]
    assert [edge.id for edge in edges] == ["valid"]


def test_paper_graph_rejects_more_than_twelve_nodes() -> None:
    nodes = [
        GraphNode(id=f"node-{index}", label=f"Node {index}", type="concept", explanation="A concept.")
        for index in range(13)
    ]

    with pytest.raises(ValidationError):
        PaperGraph(nodes=nodes)


def test_visual_filename_cannot_escape_the_visuals_directory() -> None:
    filename = _visual_filename("../outside/figure 3", 1)

    assert "/" not in filename
    assert filename == "001-outside-figure-3.webp"

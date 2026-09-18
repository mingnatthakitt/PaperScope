from __future__ import annotations

from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from paperscope.schemas import RetrievedContext


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


def retrieve_chunks(session: Session, paper_ids: list[UUID], vector: list[float], limit: int = 20) -> list[RetrievedContext]:
    statement = text(
        """
        SELECT id, paper_id, content, section_title, page_start, page_end,
               1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM paper_chunks
        WHERE paper_id IN :paper_ids AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
        """
    ).bindparams(bindparam("paper_ids", expanding=True))
    rows = session.execute(statement, {"embedding": _vector_literal(vector), "paper_ids": paper_ids, "limit": limit})
    return [
        RetrievedContext(
            id=f"chunk:{row.id}",
            paper_id=row.paper_id,
            source_type="text",
            content=row.content,
            similarity=float(row.similarity),
            section=row.section_title,
            page_start=row.page_start,
            page_end=row.page_end,
            metadata={"databaseId": str(row.id)},
        )
        for row in rows
    ]


def retrieve_visuals(session: Session, paper_ids: list[UUID], vector: list[float], limit: int = 20) -> list[RetrievedContext]:
    statement = text(
        """
        SELECT id, paper_id, visual_type, figure_label, caption,
               visual_description, visual_interpretation, page_number,
               1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM paper_visuals
        WHERE paper_id IN :paper_ids AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
        """
    ).bindparams(bindparam("paper_ids", expanding=True))
    rows = session.execute(statement, {"embedding": _vector_literal(vector), "paper_ids": paper_ids, "limit": limit})
    return [
        RetrievedContext(
            id=f"visual:{row.id}",
            paper_id=row.paper_id,
            source_type="table" if row.visual_type == "table" else "visual",
            content=f"{row.figure_label or row.visual_type}. {row.caption or ''}\n{row.visual_description}\n{row.visual_interpretation or ''}".strip(),
            similarity=float(row.similarity),
            page_start=row.page_number,
            page_end=row.page_number,
            visual_id=row.id,
            image_refs=[str(row.id)],
            metadata={"databaseId": str(row.id), "label": row.figure_label, "visualType": row.visual_type},
        )
        for row in rows
    ]


def retrieve_graph_nodes(session: Session, paper_ids: list[UUID], vector: list[float], limit: int = 20) -> list[RetrievedContext]:
    statement = text(
        """
        SELECT n.id, g.paper_id, n.label, n.node_type, n.explanation,
               n.evidence_ids,
               1 - (n.embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM paper_graph_nodes n
        JOIN paper_graphs g ON g.id = n.graph_id
        WHERE g.paper_id IN :paper_ids AND n.embedding IS NOT NULL
        ORDER BY n.embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
        """
    ).bindparams(bindparam("paper_ids", expanding=True))
    rows = session.execute(statement, {"embedding": _vector_literal(vector), "paper_ids": paper_ids, "limit": limit})
    return [
        RetrievedContext(
            id=f"graph:{row.id}",
            paper_id=row.paper_id,
            source_type="graph_node",
            content=f"{row.label}: {row.explanation}",
            similarity=float(row.similarity),
            metadata={"databaseId": str(row.id), "nodeType": row.node_type, "evidenceIds": row.evidence_ids or []},
        )
        for row in rows
    ]


def retrieve_all(session: Session, paper_ids: list[UUID], vector: list[float]) -> list[RetrievedContext]:
    candidates = retrieve_chunks(session, paper_ids, vector)
    candidates.extend(retrieve_visuals(session, paper_ids, vector))
    candidates.extend(retrieve_graph_nodes(session, paper_ids, vector))
    candidates.sort(key=lambda item: item.similarity, reverse=True)

    deduped: list[RetrievedContext] = []
    seen: set[tuple[str, str, str]] = set()
    for item in candidates:
        key = (str(item.paper_id), item.source_type, item.content[:180].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    # Comparison questions need attributable evidence from each selected paper when
    # the index contains a match. Reserve one slot per paper before filling the
    # remaining slots by global similarity.
    unique: list[RetrievedContext] = []
    selected_keys: set[tuple[str, str, str]] = set()
    for paper_id in paper_ids:
        for item in deduped:
            key = (str(item.paper_id), item.source_type, item.content[:180].lower())
            if item.paper_id == paper_id and key not in selected_keys:
                unique.append(item)
                selected_keys.add(key)
                break
    for item in deduped:
        key = (str(item.paper_id), item.source_type, item.content[:180].lower())
        if key in selected_keys:
            continue
        unique.append(item)
        selected_keys.add(key)
        if len(unique) >= 10:
            break
    return unique[:10]

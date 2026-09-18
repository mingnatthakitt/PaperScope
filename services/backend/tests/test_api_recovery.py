from uuid import uuid4

import pytest
from fastapi import HTTPException

from paperscope.main import retry_job
from paperscope.models.tables import IngestionJob, Paper


class _Session:
    def __init__(self, job, paper):
        self.job = job
        self.paper = paper
        self.commits = 0

    def get(self, model, identifier):
        if model is IngestionJob and identifier == self.job.id:
            return self.job
        if model is Paper and identifier == self.paper.id:
            return self.paper
        return None

    def commit(self):
        self.commits += 1


def _failed_records():
    paper_id = uuid4()
    job = IngestionJob(id=uuid4(), paper_id=paper_id, status="failed", stage="analyzing", attempts=3)
    paper = Paper(id=paper_id, sha256="a" * 64, original_path="paper/original.pdf", status="failed")
    return job, paper


def test_retry_reuses_failed_job_and_resets_attempts() -> None:
    job, paper = _failed_records()
    session = _Session(job, paper)

    response = retry_job(job.id, session)

    assert response.job_id == job.id
    assert job.status == "queued"
    assert job.attempts == 0
    assert paper.status == "queued"
    assert session.commits == 1


def test_retry_does_not_restart_a_ready_paper() -> None:
    job, paper = _failed_records()
    paper.status = "ready"
    session = _Session(job, paper)

    with pytest.raises(HTTPException) as error:
        retry_job(job.id, session)

    assert error.value.status_code == 409

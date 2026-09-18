from types import SimpleNamespace
from uuid import uuid4

from paperscope import worker
from paperscope.models.tables import IngestionJob, Paper
from paperscope.providers.contracts import ProviderAttemptsExhaustedError, RetryableProviderError


class _Session:
    def __init__(self, job, paper):
        self.job = job
        self.paper = paper

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def rollback(self):
        return None

    def commit(self):
        return None

    def get(self, model, identifier):
        if model is IngestionJob and identifier == self.job.id:
            return self.job
        if model is Paper and identifier == self.paper.id:
            return self.paper
        return None


def test_retryable_worker_failure_is_requeued(monkeypatch) -> None:
    job = SimpleNamespace(
        id=uuid4(),
        paper_id=uuid4(),
        attempts=1,
        status="running",
        stage="analyzing",
        locked_until=None,
        error_code=None,
        error_message=None,
        message="",
    )
    paper = SimpleNamespace(id=job.paper_id, status="processing")

    class FailingPipeline:
        def __init__(self, session):
            return None

        def process(self, job_id):
            raise RetryableProviderError("temporary provider outage")

    monkeypatch.setattr(worker, "claim_job", lambda: job)
    monkeypatch.setattr(worker, "IngestionPipeline", FailingPipeline)
    monkeypatch.setattr(worker, "SessionLocal", lambda: _Session(job, paper))

    assert worker.run_once()
    assert job.status == "queued"
    assert paper.status == "queued"
    assert job.error_code == "RetryableProviderError"
    assert job.locked_until is not None


def test_terminal_worker_failure_is_marked_failed(monkeypatch) -> None:
    job = SimpleNamespace(
        id=uuid4(),
        paper_id=uuid4(),
        attempts=3,
        status="running",
        stage="analyzing",
        locked_until=None,
        error_code=None,
        error_message=None,
        message="",
    )
    paper = SimpleNamespace(id=job.paper_id, status="processing")

    class FailingPipeline:
        def __init__(self, session):
            return None

        def process(self, job_id):
            raise RetryableProviderError("provider remains unavailable")

    monkeypatch.setattr(worker, "claim_job", lambda: job)
    monkeypatch.setattr(worker, "IngestionPipeline", FailingPipeline)
    monkeypatch.setattr(worker, "SessionLocal", lambda: _Session(job, paper))

    assert worker.run_once()
    assert job.status == "failed"
    assert paper.status == "failed"
    assert job.locked_until is None


def test_exhausted_gemini_chain_is_not_replayed_by_worker(monkeypatch) -> None:
    job = SimpleNamespace(
        id=uuid4(),
        paper_id=uuid4(),
        attempts=1,
        status="running",
        stage="analyzing",
        locked_until=None,
        error_code=None,
        error_message=None,
        message="",
    )
    paper = SimpleNamespace(id=job.paper_id, status="processing")

    class FailingPipeline:
        def __init__(self, session):
            return None

        def process(self, job_id):
            raise ProviderAttemptsExhaustedError("Gemini 3.8 -> 3.7 -> 3.6 exhausted")

    monkeypatch.setattr(worker, "claim_job", lambda: job)
    monkeypatch.setattr(worker, "IngestionPipeline", FailingPipeline)
    monkeypatch.setattr(worker, "SessionLocal", lambda: _Session(job, paper))

    assert worker.run_once()
    assert job.status == "failed"
    assert paper.status == "failed"

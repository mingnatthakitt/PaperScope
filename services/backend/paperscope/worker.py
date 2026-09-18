from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, or_, select, update

from paperscope.config import settings
from paperscope.db.session import SessionLocal
from paperscope.models.tables import IngestionJob, Paper
from paperscope.processing.ingest import IngestionPipeline
from paperscope.providers.contracts import ProviderAttemptsExhaustedError, RetryableProviderError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("paperscope.worker")


def claim_job() -> IngestionJob | None:
    with SessionLocal() as session:
        now = datetime.now(UTC)
        job = session.scalar(
            select(IngestionJob)
            .where(
                or_(
                    and_(
                        IngestionJob.status == "queued",
                        or_(IngestionJob.locked_until.is_(None), IngestionJob.locked_until < now),
                    ),
                    and_(
                        IngestionJob.status == "running",
                        or_(IngestionJob.locked_until.is_(None), IngestionJob.locked_until < now),
                    ),
                )
            )
            .order_by(IngestionJob.created_at)
            .with_for_update(skip_locked=True)
        )
        if not job:
            return None
        job.status = "running"
        job.attempts += 1
        job.locked_until = now + timedelta(seconds=settings.ingestion_lease_seconds)
        session.commit()
        session.refresh(job)
        return job


def _renew_lease(job_id, stop: threading.Event) -> None:
    interval = max(1, settings.ingestion_lease_seconds // 3)
    while not stop.wait(interval):
        try:
            with SessionLocal() as session:
                session.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == job_id, IngestionJob.status == "running")
                    .values(locked_until=datetime.now(UTC) + timedelta(seconds=settings.ingestion_lease_seconds))
                )
                session.commit()
        except Exception:
            logger.exception("Could not renew lease for ingestion job %s", job_id)


def run_once() -> bool:
    claimed = claim_job()
    if not claimed:
        return False
    claimed_id = claimed.id
    claimed_paper_id = claimed.paper_id
    logger.info("Processing paper %s", claimed_paper_id)
    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(target=_renew_lease, args=(claimed_id, stop_heartbeat), daemon=True)
    heartbeat.start()
    with SessionLocal() as session:
        try:
            IngestionPipeline(session).process(claimed_id)
        except Exception as exc:
            logger.exception("Paper %s failed", claimed_paper_id)
            session.rollback()
            job = session.get(IngestionJob, claimed_id)
            paper = session.get(Paper, claimed_paper_id)
            retryable = isinstance(exc, RetryableProviderError) and not isinstance(exc, ProviderAttemptsExhaustedError)
            if job and retryable and job.attempts < settings.ingestion_max_attempts:
                retry_delay = settings.ingestion_retry_backoff_seconds * (2 ** max(job.attempts - 1, 0))
                job.status = "queued"
                job.error_code = type(exc).__name__
                job.error_message = str(exc)[:2000]
                job.message = f"Retrying ingestion after a transient provider failure in {retry_delay}s"
                job.locked_until = datetime.now(UTC) + timedelta(seconds=retry_delay)
                if paper:
                    paper.status = "queued"
            elif job:
                job.status = "failed"
                job.stage = job.stage or "failed"
                job.error_code = type(exc).__name__
                job.error_message = str(exc)[:2000]
                job.message = "Ingestion failed. Review the error and retry."
                job.locked_until = None
            if paper:
                paper.status = "queued" if retryable and job and job.status == "queued" else "failed"
            session.commit()
        finally:
            stop_heartbeat.set()
            heartbeat.join(timeout=2)
    return True


def main() -> None:
    while True:
        if not run_once():
            time.sleep(2)


if __name__ == "__main__":
    main()

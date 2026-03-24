# -*- coding: utf-8 -*-
"""Worker runtime service helpers."""
import json
from datetime import datetime
from typing import Any

import aiofiles
from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Job, JobStatus, User, Worker
from app.services.image_storage import ensure_thumbnail

ALLOWED_WORKER_STATUSES = {"idle", "busy", "offline", "online"}
TERMINAL_JOB_STATUSES = {
    JobStatus.DONE.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
}


def normalize_worker_status(status: str) -> str:
    """Normalize worker status values for storage."""
    normalized = (status or "").strip().lower()
    if normalized not in ALLOWED_WORKER_STATUSES:
        raise HTTPException(status_code=400, detail=f"不支持的 Worker 状态: {status}")
    if normalized == "online":
        return "idle"
    return normalized


def serialize_job_for_worker(job: Job) -> dict[str, Any]:
    """Build the payload returned to a worker when a job is claimed."""
    return {
        "id": job.id,
        "user_id": job.user_id,
        "prompt": job.prompt,
        "negative_prompt": job.negative_prompt,
        "width": job.width,
        "height": job.height,
        "steps": job.steps,
        "cfg_scale": job.cfg_scale,
        "seed": job.seed,
        "sampler": job.sampler,
        "created_at": job.created_at.isoformat(),
    }


def serialize_worker_snapshot(worker: Worker) -> dict[str, Any]:
    """Build the public worker snapshot used by status pages."""
    return {
        "id": worker.id,
        "name": worker.name,
        "status": "online" if worker.is_online else "offline",
        "is_busy": worker.status == "busy",
        "current_job_id": worker.current_job_id if worker.is_online else None,
        "gpu_info": worker.gpu_info,
        "last_seen_at": worker.last_seen_at,
    }


async def get_job_or_404(job_id: str, db: AsyncSession) -> Job:
    """Load a job or raise a 404."""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


def ensure_worker_owns_job(worker: Worker, job: Job) -> None:
    """Reject updates coming from a different worker after assignment."""
    if job.worker_id and job.worker_id != worker.id:
        raise HTTPException(status_code=403, detail="该任务已被其他 Worker 领取")


async def heartbeat_worker(
    *,
    worker: Worker,
    status: str,
    current_job_id: str | None,
    gpu_info: dict[str, Any] | None,
) -> Worker:
    """Update worker liveness and runtime metadata."""
    worker.status = normalize_worker_status(status)
    worker.last_seen_at = datetime.utcnow()
    worker.current_job_id = current_job_id

    if gpu_info is not None:
        worker.gpu_info = gpu_info

    return worker


async def claim_next_job(*, worker: Worker, db: AsyncSession) -> Job | None:
    """Atomically claim the next queued job for a worker."""
    result = await db.execute(
        select(Job)
        .where(Job.status == JobStatus.QUEUED.value)
        .order_by(Job.priority.desc(), Job.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalar_one_or_none()

    if not job:
        worker.last_seen_at = datetime.utcnow()
        if worker.status != "offline":
            worker.status = "idle"
        worker.current_job_id = None
        return None

    now = datetime.utcnow()
    job.status = JobStatus.RUNNING.value
    job.started_at = now
    job.worker_id = worker.id

    worker.status = "busy"
    worker.current_job_id = job.id
    worker.last_seen_at = now

    await db.commit()
    return job


async def update_job_status_for_worker(
    *,
    worker: Worker,
    job_id: str,
    status: str,
    db: AsyncSession,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Apply a worker-reported job status update."""
    normalized_status = (status or "").strip().lower()
    if normalized_status not in {item.value for item in JobStatus}:
        raise HTTPException(status_code=400, detail=f"不支持的任务状态: {status}")

    job = await get_job_or_404(job_id, db)
    ensure_worker_owns_job(worker, job)

    old_status = job.status
    job.status = normalized_status

    now = datetime.utcnow()
    if normalized_status == JobStatus.RUNNING.value:
        job.started_at = now
        job.worker_id = worker.id
        worker.status = "busy"
        worker.current_job_id = job.id
    elif normalized_status in TERMINAL_JOB_STATUSES:
        job.finished_at = now
        if error_message:
            job.error_message = error_message
        if worker.current_job_id == job.id:
            worker.current_job_id = None
        if worker.status != "offline":
            worker.status = "idle"

    worker.last_seen_at = now

    return {"success": True, "old_status": old_status, "new_status": job.status}


async def upload_job_result_for_worker(
    *,
    worker: Worker,
    job_id: str,
    image: UploadFile,
    metadata: str,
    db: AsyncSession,
) -> dict[str, Any]:
    """Persist a worker-generated image and finalize the job."""
    job = await get_job_or_404(job_id, db)
    ensure_worker_owns_job(worker, job)

    try:
        parsed_metadata = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        parsed_metadata = {}

    today = datetime.now().strftime("%Y-%m-%d")
    save_dir = settings.STORAGE_ROOT / str(job.user_id) / today
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{job_id}.png"

    async with aiofiles.open(save_path, "wb") as output_file:
        content = await image.read()
        await output_file.write(content)

    now = datetime.utcnow()
    job.image_path = str(save_path.relative_to(settings.STORAGE_ROOT))
    job.result_metadata = parsed_metadata
    job.status = JobStatus.DONE.value
    job.worker_id = worker.id
    job.error_message = None
    job.finished_at = now
    if not job.started_at:
        job.started_at = now
    ensure_thumbnail(job.image_path)

    result = await db.execute(select(User).where(User.id == job.user_id))
    user = result.scalar_one_or_none()
    if user:
        user.today_used_count = (user.today_used_count or 0) + 1
        user.total_generations = (user.total_generations or 0) + 1

    worker.last_seen_at = now
    worker.current_job_id = None
    if worker.status != "offline":
        worker.status = "idle"

    return {"success": True, "image_path": job.image_path}

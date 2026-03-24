# -*- coding: utf-8 -*-
"""Reusable worker-facing API."""
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import verify_worker_auth
from app.models import Worker, get_db
from app.services.worker_runtime import (
    claim_next_job,
    heartbeat_worker,
    serialize_job_for_worker,
    update_job_status_for_worker,
    upload_job_result_for_worker,
)

router = APIRouter()


class WorkerHeartbeatRequest(BaseModel):
    """Worker heartbeat payload."""
    status: str = Field(..., description="idle | busy | offline | online")
    current_job_id: Optional[str] = None
    gpu_info: Optional[dict[str, Any]] = None


class WorkerHeartbeatResponse(BaseModel):
    """Worker heartbeat response."""
    success: bool
    message: str = "OK"


class WorkerJobStatusRequest(BaseModel):
    """Worker job status payload."""
    status: str = Field(..., description="queued | running | done | failed | cancelled")
    error_message: Optional[str] = None


@router.post("/heartbeat", response_model=WorkerHeartbeatResponse)
async def worker_heartbeat(
    payload: WorkerHeartbeatRequest,
    worker: Worker = Depends(verify_worker_auth),
    _db: AsyncSession = Depends(get_db),
):
    """Update a worker heartbeat using auth headers as identity."""
    await heartbeat_worker(
        worker=worker,
        status=payload.status,
        current_job_id=payload.current_job_id,
        gpu_info=payload.gpu_info,
    )
    return WorkerHeartbeatResponse(success=True)


@router.post("/jobs/claim")
async def claim_worker_job(
    worker: Worker = Depends(verify_worker_auth),
    db: AsyncSession = Depends(get_db),
):
    """Claim the next queued job for the authenticated worker."""
    job = await claim_next_job(worker=worker, db=db)
    if not job:
        return Response(status_code=204)
    return serialize_job_for_worker(job)


@router.patch("/jobs/{job_id}/status")
async def worker_update_job_status(
    job_id: str,
    payload: WorkerJobStatusRequest,
    worker: Worker = Depends(verify_worker_auth),
    db: AsyncSession = Depends(get_db),
):
    """Update job status from a worker through the unified API."""
    return await update_job_status_for_worker(
        worker=worker,
        job_id=job_id,
        status=payload.status,
        error_message=payload.error_message,
        db=db,
    )


@router.post("/jobs/{job_id}/result")
async def worker_upload_job_result(
    job_id: str,
    image: UploadFile = File(...),
    metadata: str = Form(default="{}"),
    worker: Worker = Depends(verify_worker_auth),
    db: AsyncSession = Depends(get_db),
):
    """Upload the generated image and finalize the job."""
    return await upload_job_result_for_worker(
        worker=worker,
        job_id=job_id,
        image=image,
        metadata=metadata,
        db=db,
    )

# -*- coding: utf-8 -*-
"""Worker API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from app.models import get_db, Worker
from app.api.deps import verify_worker_auth, get_current_admin
from app.services.worker_runtime import (
    claim_next_job,
    heartbeat_worker,
    serialize_job_for_worker,
    serialize_worker_snapshot,
)

router = APIRouter()


class HeartbeatRequest(BaseModel):
    """心跳请求"""
    worker_id: str
    status: str  # idle / busy / offline / online
    current_job_id: Optional[str] = None
    gpu_info: Optional[dict] = None


class HeartbeatResponse(BaseModel):
    """心跳响应"""
    success: bool
    message: str = "OK"


@router.post("/heartbeat", response_model=HeartbeatResponse)
async def heartbeat(
    data: HeartbeatRequest,
    worker: Worker = Depends(verify_worker_auth),
    _db: AsyncSession = Depends(get_db),
):
    """
    Worker 心跳
    
    Worker 定期发送心跳，更新状态和 GPU 信息
    """
    if worker.id != data.worker_id:
        raise HTTPException(status_code=403, detail="Worker ID 不匹配")

    await heartbeat_worker(
        worker=worker,
        status=data.status,
        current_job_id=data.current_job_id,
        gpu_info=data.gpu_info,
    )

    return HeartbeatResponse(success=True)


@router.get("/{worker_id}/next-job")
async def get_next_job(
    worker_id: str,
    worker: Worker = Depends(verify_worker_auth),
    db: AsyncSession = Depends(get_db),
):
    """
    拉取下一个待处理任务（原子性领取）
    
    使用 SELECT FOR UPDATE 锁定任务，确保同一任务不会被多个 Worker 领取
    """
    # 检查 Worker 是否匹配
    if worker.id != worker_id:
        raise HTTPException(status_code=403, detail="Worker ID 不匹配")
    
    job = await claim_next_job(worker=worker, db=db)
    if not job:
        # 无任务，返回 204 No Content
        from fastapi.responses import Response
        return Response(status_code=204)

    return serialize_job_for_worker(job)


@router.get("")
async def list_workers(
    db: AsyncSession = Depends(get_db),
):
    """获取所有 Worker 状态（公开接口，用于显示服务状态）"""
    result = await db.execute(select(Worker))
    workers = result.scalars().all()
    
    return {
        "workers": [
            serialize_worker_snapshot(w)
            for w in workers
        ],
        "online_count": sum(1 for w in workers if w.is_online),
        "total_count": len(workers),
    }


@router.delete("/{worker_id}")
async def delete_worker(
    worker_id: str,
    admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """删除 Worker 记录（仅管理员，且只能删除离线的 Worker）"""
    result = await db.execute(select(Worker).where(Worker.id == worker_id))
    worker = result.scalar_one_or_none()
    
    if not worker:
        raise HTTPException(status_code=404, detail="Worker 不存在")
    
    if worker.is_online:
        raise HTTPException(status_code=400, detail="无法删除在线的 Worker")
    
    await db.delete(worker)
    await db.commit()
    
    return {"success": True, "message": f"已删除 Worker: {worker_id}"}




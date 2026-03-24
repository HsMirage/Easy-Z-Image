# -*- coding: utf-8 -*-
"""任务 API"""
import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from pydantic import BaseModel, Field

from app.models import get_db, User, Job, JobStatus, Worker
from app.config import settings
from app.api.deps import get_current_user, verify_worker_auth
from app.services.image_storage import build_image_url, build_thumbnail_url
from app.services.worker_runtime import (
    update_job_status_for_worker,
    upload_job_result_for_worker,
)

router = APIRouter()


class JobCreate(BaseModel):
    """创建任务请求"""
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = Field(default="", max_length=1000)
    width: int = Field(default=1024, ge=256, le=settings.MAX_WIDTH)
    height: int = Field(default=1024, ge=256, le=settings.MAX_HEIGHT)
    steps: int = Field(default=9, ge=4, le=30)
    seed: int = Field(default=-1)


class JobResponse(BaseModel):
    """任务响应"""
    id: str
    status: str
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None
    queue_position: Optional[int] = None
    queue_overload: bool = False
    is_public: bool = False
    is_anonymous: bool = True
    publish_status: str = "private"
    is_nsfw: bool = False


class JobStatusUpdate(BaseModel):
    """任务状态更新"""
    status: str
    error_message: Optional[str] = None


@router.post("", response_model=JobResponse)
async def create_job(
    job_data: JobCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """提交生图任务"""
    
    # 检查是否有在线 Worker
    result = await db.execute(select(Worker))
    workers = result.scalars().all()
    online_workers = [w for w in workers if w.is_online]
    
    if not online_workers:
        raise HTTPException(status_code=503, detail="生图服务当前离线，请稍后再试")
    
    # 管理员跳过队列限制检查
    if not user.is_admin:
        # 检查用户是否有待处理任务
        result = await db.execute(
            select(func.count(Job.id)).where(
                and_(
                    Job.user_id == user.id,
                    Job.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value])
                )
            )
        )
        pending_count = result.scalar()
        
        if pending_count >= 1:
            raise HTTPException(status_code=429, detail="您已有任务在处理中，请等待完成后再提交")
        
        # 检查配额
        if not user.can_submit_job():
            raise HTTPException(status_code=429, detail=f"今日配额已用完（{user.daily_quota}张/天）")
        
        # 检查队列长度
        result = await db.execute(
            select(func.count(Job.id)).where(Job.status == JobStatus.QUEUED.value)
        )
        queue_length = result.scalar()
        
        if queue_length >= settings.HARD_QUEUE_LIMIT:
            raise HTTPException(status_code=503, detail="系统繁忙，请稍后再试")
    
    # 检查队列长度（用于提示）
    result = await db.execute(
        select(func.count(Job.id)).where(Job.status == JobStatus.QUEUED.value)
    )
    queue_length = result.scalar()
    queue_overload = queue_length >= settings.MAX_QUEUE_LENGTH
    
    # 管理员任务优先级更高（插队）
    priority = 10 if user.is_admin else 0
    
    # 创建任务
    job = Job(
        id=str(uuid.uuid4()),
        user_id=user.id,
        prompt=job_data.prompt.strip(),
        negative_prompt=job_data.negative_prompt.strip() if job_data.negative_prompt else None,
        width=job_data.width,
        height=job_data.height,
        steps=job_data.steps,
        seed=job_data.seed,
        status=JobStatus.QUEUED.value,
        priority=priority,
    )
    
    db.add(job)
    await db.flush()
    
    return JobResponse(
        id=job.id,
        status=job.status,
        prompt=job.prompt,
        width=job.width,
        height=job.height,
        steps=job.steps,
        seed=job.seed,
        created_at=job.created_at,
        queue_position=queue_length + 1,
        queue_overload=queue_overload,
        publish_status=job.public_status or "private",
        is_nsfw=job.is_nsfw or False,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取任务详情"""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 非管理员只能查看自己的任务
    if not user.is_admin and job.user_id != user.id:
        raise HTTPException(status_code=403, detail="无权查看此任务")
    
    # 计算队列位置
    queue_position = None
    if job.status == JobStatus.QUEUED.value:
        result = await db.execute(
            select(func.count(Job.id)).where(
                and_(
                    Job.status == JobStatus.QUEUED.value,
                    Job.created_at < job.created_at
                )
            )
        )
        queue_position = result.scalar() + 1
    
    image_url = build_image_url(job.image_path)
    
    return JobResponse(
        id=job.id,
        status=job.status,
        prompt=job.prompt,
        width=job.width,
        height=job.height,
        steps=job.steps,
        seed=job.seed if job.seed >= 0 else (job.result_metadata or {}).get("seed", -1),
        image_url=image_url,
        thumbnail_url=build_thumbnail_url(job.image_path),
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        elapsed_seconds=job.elapsed_seconds,
        queue_position=queue_position,
        is_public=job.is_public or False,
        is_anonymous=job.is_anonymous if job.is_anonymous is not None else True,
        publish_status=job.public_status or "private",
        is_nsfw=job.is_nsfw or False,
    )


@router.get("")
async def list_jobs(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    page: int = 1,
    limit: int = 20,
):
    """获取用户的任务列表（排除已删除、24小时前已取消的、1小时前失败的）"""
    offset = (page - 1) * limit
    
    # 时间阈值
    cancel_threshold = datetime.utcnow() - timedelta(hours=24)  # 24小时前
    failed_threshold = datetime.utcnow() - timedelta(hours=1)   # 1小时前
    
    # 基础过滤条件：用户的任务 + 未删除 + 排除24小时前已取消的 + 排除1小时前失败的
    base_filter = (
        (Job.user_id == user.id) &
        ((Job.is_deleted == False) | (Job.is_deleted.is_(None))) &
        ~((Job.status == JobStatus.CANCELLED.value) & (Job.finished_at < cancel_threshold)) &
        ~((Job.status == JobStatus.FAILED.value) & (Job.finished_at < failed_threshold))
    )
    
    # 获取总数
    count_result = await db.execute(
        select(func.count(Job.id)).where(base_filter)
    )
    total = count_result.scalar() or 0
    
    # 获取分页数据
    result = await db.execute(
        select(Job)
        .where(base_filter)
        .order_by(Job.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    jobs = result.scalars().all()
    
    return {
        "jobs": [
            {
                "id": job.id,
                "status": job.status,
                "prompt": job.prompt,
                "width": job.width,
                "height": job.height,
                "image_url": build_image_url(job.image_path),
                "thumbnail_url": build_thumbnail_url(job.image_path),
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "is_public": job.is_public or False,
                "is_anonymous": job.is_anonymous if job.is_anonymous is not None else True,
                "publish_status": job.public_status or "private",
                "is_nsfw": job.is_nsfw or False,
            }
            for job in jobs
        ],
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit if total else 0,
    }


@router.patch("/{job_id}/status")
async def update_job_status(
    job_id: str,
    status_update: JobStatusUpdate,
    worker: Worker = Depends(verify_worker_auth),
    db: AsyncSession = Depends(get_db),
):
    """Worker 更新任务状态"""
    return await update_job_status_for_worker(
        worker=worker,
        job_id=job_id,
        status=status_update.status,
        error_message=status_update.error_message,
        db=db,
    )


@router.post("/{job_id}/result")
async def upload_job_result(
    job_id: str,
    image: UploadFile = File(...),
    metadata: str = Form(default="{}"),
    worker: Worker = Depends(verify_worker_auth),
    db: AsyncSession = Depends(get_db),
):
    """Worker 上传生成结果"""
    return await upload_job_result_for_worker(
        worker=worker,
        job_id=job_id,
        image=image,
        metadata=metadata,
        db=db,
    )


@router.get("/{job_id}/image")
async def get_job_image(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """获取任务图片"""
    from fastapi.responses import FileResponse
    
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job or not job.image_path:
        raise HTTPException(status_code=404, detail="图片不存在")
    
    file_path = settings.STORAGE_ROOT / job.image_path
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="图片文件不存在")
    
    return FileResponse(file_path, media_type="image/png")


class PublishRequest(BaseModel):
    """发布请求"""
    is_anonymous: bool = True  # 默认匿名
    is_nsfw: bool = False


@router.post("/{job_id}/publish")
async def publish_job(
    job_id: str,
    req: PublishRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    发布作品到广场
    
    只有作品所有者才能发布
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="作品不存在")
    
    if job.user_id != user.id:
        raise HTTPException(status_code=403, detail="只能发布自己的作品")
    
    if job.status != JobStatus.DONE.value:
        raise HTTPException(status_code=400, detail="只能发布已完成的作品")
    
    if job.public_status == "pending":
        raise HTTPException(status_code=400, detail="作品已提交审核，请等待管理员处理")
    
    if job.is_public or job.public_status == "approved":
        raise HTTPException(status_code=400, detail="作品已发布")

    job.is_public = False
    job.public_status = "pending"
    job.is_anonymous = req.is_anonymous
    job.is_nsfw = req.is_nsfw
    job.reviewed_at = None
    await db.commit()
    
    return {
        "success": True,
        "message": "作品已提交审核，审核通过后会出现在作品广场",
        "is_public": job.is_public,
        "publish_status": job.public_status,
        "is_nsfw": job.is_nsfw,
    }


@router.post("/{job_id}/unpublish")
async def unpublish_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    取消发布（从广场移除）
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="作品不存在")
    
    if job.user_id != user.id:
        raise HTTPException(status_code=403, detail="只能操作自己的作品")
    
    if not job.is_public and job.public_status != "pending":
        raise HTTPException(status_code=400, detail="作品未发布，也未在审核中")
    
    was_pending = job.public_status == "pending"
    job.is_public = False
    job.public_status = "private"
    job.reviewed_at = None
    await db.commit()
    
    return {
        "success": True,
        "message": "已撤回审核申请" if was_pending else "已从广场移除",
        "is_public": job.is_public,
        "publish_status": job.public_status,
    }


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    用户删除作品（软删除，不删除实际文件）
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="作品不存在")
    
    if job.user_id != user.id:
        raise HTTPException(status_code=403, detail="只能删除自己的作品")
    
    if job.is_deleted:
        raise HTTPException(status_code=400, detail="作品已删除")
    
    # 软删除
    job.is_deleted = True
    job.deleted_at = datetime.utcnow()
    # 同时从广场移除
    job.is_public = False
    job.public_status = "private"
    await db.commit()
    
    return {"success": True, "message": "作品已删除"}


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    取消自己的任务（仅限排队中或生成中）
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if job.user_id != user.id:
        raise HTTPException(status_code=403, detail="只能取消自己的任务")
    
    if job.status not in [JobStatus.QUEUED.value, JobStatus.RUNNING.value]:
        raise HTTPException(status_code=400, detail="只能取消排队中或生成中的任务")
    
    job.status = JobStatus.CANCELLED.value
    await db.commit()
    
    return {"success": True, "message": "任务已取消"}

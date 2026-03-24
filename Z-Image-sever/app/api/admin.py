# -*- coding: utf-8 -*-
"""管理员 API"""
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from pydantic import BaseModel
from typing import List, Optional, Literal

from app.models import get_db, User, Job, Worker
from app.api.deps import get_current_admin
from app.services.image_storage import build_image_url, build_thumbnail_url

router = APIRouter()


class AdminStats(BaseModel):
    total_users: int
    total_jobs: int
    total_completed: int
    total_failed: int
    today_jobs: int
    online_workers: int
    total_workers: int


class AdminUserResponse(BaseModel):
    id: int
    username: str
    nickname: Optional[str]
    avatar_url: Optional[str]
    trust_level: int
    is_admin: bool
    is_active: bool
    daily_quota: int
    quota_override: Optional[int]
    today_used_count: int
    total_generations: int
    created_at: datetime


class UpdateUserQuotaRequest(BaseModel):
    daily_quota: int


class AdminJobResponse(BaseModel):
    id: str
    user: dict
    status: str
    prompt: str
    width: int
    height: int
    worker_id: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    finished_at: Optional[datetime]
    elapsed_seconds: Optional[float]
    image_url: Optional[str]
    thumbnail_url: Optional[str] = None
    is_public: bool = False
    is_deleted: bool = False
    publish_status: str = "private"
    is_nsfw: bool = False


class AdminJobReviewRequest(BaseModel):
    is_nsfw: bool = False


class AdminWorkerResponse(BaseModel):
    id: str
    name: str
    status: str
    is_busy: bool
    current_job_id: Optional[str]
    gpu_info: Optional[dict]
    last_seen_at: datetime


@router.get("/stats", response_model=AdminStats)
async def get_admin_stats(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """获取管理统计数据"""
    # 总用户数
    total_users = await db.scalar(select(func.count(User.id)))
    
    # 任务统计
    total_jobs = await db.scalar(select(func.count(Job.id)))
    total_completed = await db.scalar(
        select(func.count(Job.id)).where(Job.status == "done")
    )
    total_failed = await db.scalar(
        select(func.count(Job.id)).where(Job.status == "failed")
    )
    
    # 今日任务
    today = date.today()
    today_jobs = await db.scalar(
        select(func.count(Job.id)).where(
            func.date(Job.created_at) == today
        )
    )
    
    # Worker 统计 (使用 is_online 属性基于心跳时间计算)
    workers_result = await db.execute(select(Worker))
    workers = workers_result.scalars().all()
    total_workers = len(workers)
    online_workers = sum(1 for w in workers if w.is_online)
    
    return AdminStats(
        total_users=total_users or 0,
        total_jobs=total_jobs or 0,
        total_completed=total_completed or 0,
        total_failed=total_failed or 0,
        today_jobs=today_jobs or 0,
        online_workers=online_workers,
        total_workers=total_workers,
    )


@router.get("/users")
async def list_users(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=10000),
):
    """获取用户列表"""
    offset = (page - 1) * limit
    
    result = await db.execute(
        select(User).order_by(User.created_at.desc()).offset(offset).limit(limit)
    )
    users = result.scalars().all()
    
    total = await db.scalar(select(func.count(User.id)))
    
    return {
        "users": [
            AdminUserResponse(
                id=u.id,
                username=u.username,
                nickname=u.nickname,
                avatar_url=u.avatar_url,
                trust_level=u.trust_level,
                is_admin=u.is_admin,
                is_active=u.is_active,
                daily_quota=u.daily_quota,
                quota_override=u.quota_override,
                today_used_count=u.today_used_count,
                total_generations=u.total_generations,
                created_at=u.created_at,
            )
            for u in users
        ],
        "total": total or 0,
        "page": page,
        "total_pages": (total or 0 + limit - 1) // limit,
    }


@router.post("/users/{user_id}/ban")
async def ban_user(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """封禁用户"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    if user.is_admin:
        raise HTTPException(status_code=400, detail="不能封禁管理员")
    
    user.is_active = False
    await db.commit()
    
    return {"success": True, "message": "用户已封禁"}


@router.post("/users/{user_id}/unban")
async def unban_user(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """解封用户"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    user.is_active = True
    await db.commit()
    
    return {"success": True, "message": "用户已解封"}


@router.post("/users/{user_id}/quota")
async def update_user_quota(
    user_id: int,
    payload: UpdateUserQuotaRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """设置用户自定义每日配额"""
    if payload.daily_quota < 0 or payload.daily_quota > 10000:
        raise HTTPException(status_code=400, detail="配额必须在 0 到 10000 之间")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if user.is_admin:
        raise HTTPException(status_code=400, detail="不能修改管理员配额")

    user.quota_override = payload.daily_quota
    user.daily_quota = payload.daily_quota
    await db.commit()

    return {
        "success": True,
        "message": "用户配额已更新",
        "daily_quota": user.daily_quota,
        "quota_override": user.quota_override,
    }


@router.post("/users/{user_id}/quota/reset")
async def reset_user_quota(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """恢复用户默认每日配额"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if user.is_admin:
        raise HTTPException(status_code=400, detail="不能修改管理员配额")

    user.quota_override = None
    user.update_quota_by_trust_level()
    await db.commit()

    return {
        "success": True,
        "message": "用户配额已恢复默认",
        "daily_quota": user.daily_quota,
        "quota_override": user.quota_override,
    }


@router.get("/users/{user_id}/jobs")
async def get_user_jobs(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    """获取指定用户的所有作品"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    offset = (page - 1) * limit
    
    result = await db.execute(
        select(Job)
        .where(Job.user_id == user_id)
        .order_by(Job.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    jobs = result.scalars().all()
    
    total = await db.scalar(
        select(func.count(Job.id)).where(Job.user_id == user_id)
    )
    
    return {
        "user": {
            "id": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "avatar_url": user.avatar_url,
        },
        "jobs": [
            {
                "id": j.id,
                "status": j.status,
                "prompt": j.prompt,
                "width": j.width,
                "height": j.height,
                "created_at": j.created_at,
                "image_url": build_image_url(j.image_path) if j.status == "done" else None,
                "thumbnail_url": build_thumbnail_url(j.image_path) if j.status == "done" else None,
            }
            for j in jobs
        ],
        "total": total or 0,
        "page": page,
        "total_pages": ((total or 0) + limit - 1) // limit,
    }


@router.get("/jobs")
async def list_jobs(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None, description="搜索提示词或用户名"),
    sort_by: Literal["created_at", "finished_at"] = Query(default="created_at"),
    sort_order: Literal["asc", "desc"] = Query(default="desc"),
):
    """获取任务列表（支持搜索和排序）"""
    offset = (page - 1) * limit
    
    # 基础查询
    query = select(Job)
    count_query = select(func.count(Job.id))
    
    # 状态筛选
    if status:
        query = query.where(Job.status == status)
        count_query = count_query.where(Job.status == status)
    
    # 搜索（提示词或用户名）
    if search:
        # 获取匹配用户名的用户ID
        user_result = await db.execute(
            select(User.id).where(
                or_(
                    User.username.ilike(f"%{search}%"),
                    User.nickname.ilike(f"%{search}%")
                )
            )
        )
        matching_user_ids = [u for u in user_result.scalars().all()]
        
        # 搜索提示词或用户
        search_condition = or_(
            Job.prompt.ilike(f"%{search}%"),
            Job.user_id.in_(matching_user_ids) if matching_user_ids else False
        )
        query = query.where(search_condition)
        count_query = count_query.where(search_condition)
    
    # 排序
    sort_column = Job.created_at if sort_by == "created_at" else Job.finished_at
    if sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())
    
    result = await db.execute(query.offset(offset).limit(limit))
    jobs = result.scalars().all()
    
    # 获取用户信息
    user_ids = list(set(j.user_id for j in jobs))
    if user_ids:
        users_result = await db.execute(
            select(User).where(User.id.in_(user_ids))
        )
        users_map = {u.id: u for u in users_result.scalars().all()}
    else:
        users_map = {}
    
    total = await db.scalar(count_query)
    
    return {
        "jobs": [
            AdminJobResponse(
                id=j.id,
                user={
                    "id": j.user_id, 
                    "username": users_map[j.user_id].username if j.user_id in users_map else "unknown",
                    "nickname": users_map[j.user_id].nickname if j.user_id in users_map else None,
                    "avatar_url": users_map[j.user_id].avatar_url if j.user_id in users_map else None,
                },
                status=j.status,
                prompt=j.prompt,
                width=j.width,
                height=j.height,
                worker_id=j.worker_id,
                error_message=j.error_message,
                created_at=j.created_at,
                finished_at=j.finished_at,
                elapsed_seconds=j.elapsed_seconds,
                image_url=build_image_url(j.image_path) if j.status == "done" else None,
                thumbnail_url=build_thumbnail_url(j.image_path) if j.status == "done" else None,
                is_public=j.is_public or False,
                is_deleted=j.is_deleted or False,
                publish_status=j.public_status or "private",
                is_nsfw=j.is_nsfw or False,
            )
            for j in jobs
        ],
        "total": total or 0,
        "page": page,
        "total_pages": ((total or 0) + limit - 1) // limit,
    }


@router.get("/workers")
async def list_workers(
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """获取 Worker 列表"""
    result = await db.execute(select(Worker).order_by(Worker.last_seen_at.desc()))
    workers = result.scalars().all()
    
    return {
        "workers": [
            AdminWorkerResponse(
                id=w.id,
                name=w.name,
                status="online" if w.is_online else "offline",  # 使用 is_online 计算
                is_busy=bool(w.current_job_id) and w.is_online,
                current_job_id=w.current_job_id if w.is_online else None,
                gpu_info=w.gpu_info,
                last_seen_at=w.last_seen_at,
            )
            for w in workers
        ]
    }


@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """重试失败的任务"""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if job.status != "failed":
        raise HTTPException(status_code=400, detail="只能重试失败的任务")
    
    job.status = "queued"
    job.error_message = None
    job.retry_count = (job.retry_count or 0) + 1
    await db.commit()
    
    return {"success": True, "message": "任务已重新排队"}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """取消任务"""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if job.status not in ["queued", "running"]:
        raise HTTPException(status_code=400, detail="只能取消排队中或运行中的任务")
    
    job.status = "cancelled"
    await db.commit()
    
    return {"success": True, "message": "任务已取消"}


@router.post("/jobs/{job_id}/unpublish")
async def admin_unpublish_job(
    job_id: str,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """管理员取消发布作品（从广场移除）"""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="作品不存在")
    
    if not job.is_public:
        raise HTTPException(status_code=400, detail="作品未发布")
    
    job.is_public = False
    job.public_status = "private"
    await db.commit()
    
    return {"success": True, "message": "已从广场移除"}


@router.post("/jobs/{job_id}/approve")
async def approve_job_publish(
    job_id: str,
    payload: AdminJobReviewRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """审核通过作品发布"""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="作品不存在")

    if job.status != "done" or not job.image_path:
        raise HTTPException(status_code=400, detail="只能审核已完成的作品")

    if job.public_status != "pending":
        raise HTTPException(status_code=400, detail="该作品当前不在待审核状态")

    job.is_public = True
    job.public_status = "approved"
    job.is_nsfw = payload.is_nsfw
    job.reviewed_at = datetime.utcnow()
    await db.commit()

    return {
        "success": True,
        "message": "作品已审核通过",
        "is_public": job.is_public,
        "publish_status": job.public_status,
        "is_nsfw": job.is_nsfw,
    }


@router.post("/jobs/{job_id}/reject")
async def reject_job_publish(
    job_id: str,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """拒绝作品发布"""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="作品不存在")

    if job.public_status != "pending":
        raise HTTPException(status_code=400, detail="该作品当前不在待审核状态")

    job.is_public = False
    job.public_status = "rejected"
    job.reviewed_at = datetime.utcnow()
    await db.commit()

    return {
        "success": True,
        "message": "作品已拒绝发布",
        "is_public": job.is_public,
        "publish_status": job.public_status,
    }

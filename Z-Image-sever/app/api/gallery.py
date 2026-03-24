# -*- coding: utf-8 -*-
"""画廊 API"""
from typing import Literal
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from sqlalchemy.orm import selectinload

from app.models import get_db, Job, JobStatus, User, Like, Comment
from app.services.image_storage import build_image_url, build_thumbnail_url

router = APIRouter()


@router.get("")
async def get_gallery(
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    sort_by: Literal["time", "likes", "comments"] = Query(default="time"),
):
    """
    获取公共画廊
    
    只展示手动发布到广场的作品 (is_public=True)
    支持按时间/点赞/评论排序
    """
    offset = (page - 1) * limit
    
    # 基础查询条件
    base_condition = (
        (Job.status == JobStatus.DONE.value) &
        (Job.image_path.isnot(None)) &
        (Job.is_public == True) &
        (Job.public_status == "approved")
    )
    
    # 根据排序方式构建查询
    if sort_by == "likes":
        # 按点赞数排序 - 使用子查询
        like_count_subq = (
            select(Like.job_id, func.count(Like.id).label("like_count"))
            .group_by(Like.job_id)
            .subquery()
        )
        query = (
            select(Job, User, func.coalesce(like_count_subq.c.like_count, 0).label("like_count"))
            .join(User, Job.user_id == User.id)
            .outerjoin(like_count_subq, Job.id == like_count_subq.c.job_id)
            .where(base_condition)
            .order_by(desc("like_count"), Job.finished_at.desc())
        )
    elif sort_by == "comments":
        # 按评论数排序
        comment_count_subq = (
            select(Comment.job_id, func.count(Comment.id).label("comment_count"))
            .group_by(Comment.job_id)
            .subquery()
        )
        query = (
            select(Job, User, func.coalesce(comment_count_subq.c.comment_count, 0).label("comment_count"))
            .join(User, Job.user_id == User.id)
            .outerjoin(comment_count_subq, Job.id == comment_count_subq.c.job_id)
            .where(base_condition)
            .order_by(desc("comment_count"), Job.finished_at.desc())
        )
    else:
        # 按时间排序（默认）
        query = (
            select(Job, User)
            .join(User, Job.user_id == User.id)
            .where(base_condition)
            .order_by(Job.finished_at.desc())
        )
    
    result = await db.execute(query.limit(limit).offset(offset))
    rows = result.all()
    
    # 获取 job_ids
    job_ids = [row[0].id for row in rows]
    
    # 批量获取点赞数
    if job_ids:
        like_counts_result = await db.execute(
            select(Like.job_id, func.count(Like.id))
            .where(Like.job_id.in_(job_ids))
            .group_by(Like.job_id)
        )
        like_counts = {job_id: count for job_id, count in like_counts_result.all()}
        
        # 批量获取评论数
        comment_counts_result = await db.execute(
            select(Comment.job_id, func.count(Comment.id))
            .where(Comment.job_id.in_(job_ids))
            .group_by(Comment.job_id)
        )
        comment_counts = {job_id: count for job_id, count in comment_counts_result.all()}
    else:
        like_counts = {}
        comment_counts = {}
    
    # 统计总数
    count_result = await db.execute(
        select(func.count(Job.id)).where(base_condition)
    )
    total = count_result.scalar()
    
    return {
        "items": [
            {
                "id": row[0].id,
                "image_url": build_image_url(row[0].image_path),
                "thumbnail_url": build_thumbnail_url(row[0].image_path),
                "prompt": row[0].prompt,
                "width": row[0].width,
                "height": row[0].height,
                "author": None if row[0].is_anonymous else {
                    "username": row[1].username,
                    "nickname": row[1].nickname,
                    "avatar_url": row[1].avatar_url,
                },
                "created_at": row[0].finished_at,
                "seed": (row[0].result_metadata or {}).get("seed"),
                "like_count": like_counts.get(row[0].id, 0),
                "comment_count": comment_counts.get(row[0].id, 0),
                "is_nsfw": row[0].is_nsfw or False,
            }
            for row in rows
        ],
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit if total else 0,
    }


@router.get("/stats")
async def get_gallery_stats(
    db: AsyncSession = Depends(get_db),
):
    """获取画廊统计"""
    # 总生成数
    result = await db.execute(
        select(func.count(Job.id)).where(Job.status == JobStatus.DONE.value)
    )
    total_images = result.scalar()
    
    # 总用户数
    result = await db.execute(select(func.count(User.id)))
    total_users = result.scalar()
    
    # 今日生成数
    from datetime import date
    today = date.today()
    result = await db.execute(
        select(func.count(Job.id))
        .where(Job.status == JobStatus.DONE.value)
        .where(func.date(Job.finished_at) == today)
    )
    today_images = result.scalar()
    
    return {
        "total_images": total_images,
        "total_users": total_users,
        "today_images": today_images,
    }

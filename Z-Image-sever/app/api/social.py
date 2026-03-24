# -*- coding: utf-8 -*-
"""社交功能 API：点赞和评论"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from pydantic import BaseModel, Field

from app.models import get_db, User, Job, Like, Comment
from app.api.deps import get_current_user

router = APIRouter()


# ==================== 点赞 ====================

class LikeResponse(BaseModel):
    liked: bool
    like_count: int


@router.post("/jobs/{job_id}/like", response_model=LikeResponse)
async def toggle_like(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """点赞/取消点赞"""
    # 检查作品是否存在且已发布
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="作品不存在")
    if not job.is_public:
        raise HTTPException(status_code=400, detail="只能对已发布的作品点赞")
    
    # 检查是否已点赞
    result = await db.execute(
        select(Like).where(Like.user_id == user.id, Like.job_id == job_id)
    )
    existing_like = result.scalar_one_or_none()
    
    if existing_like:
        # 取消点赞
        await db.delete(existing_like)
        liked = False
    else:
        # 添加点赞
        new_like = Like(user_id=user.id, job_id=job_id)
        db.add(new_like)
        liked = True
    
    await db.commit()
    
    # 获取点赞数
    result = await db.execute(
        select(func.count(Like.id)).where(Like.job_id == job_id)
    )
    like_count = result.scalar() or 0
    
    return LikeResponse(liked=liked, like_count=like_count)


@router.get("/jobs/{job_id}/like-status")
async def get_like_status(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取点赞状态"""
    # 检查是否已点赞
    result = await db.execute(
        select(Like).where(Like.user_id == user.id, Like.job_id == job_id)
    )
    liked = result.scalar_one_or_none() is not None
    
    # 获取点赞数
    result = await db.execute(
        select(func.count(Like.id)).where(Like.job_id == job_id)
    )
    like_count = result.scalar() or 0
    
    return {"liked": liked, "like_count": like_count}


# ==================== 评论 ====================

class CommentCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=500)


class CommentResponse(BaseModel):
    id: int
    content: str
    created_at: datetime
    user: dict  # {username, nickname, avatar_url}


class CommentsListResponse(BaseModel):
    comments: List[CommentResponse]
    total: int


@router.post("/jobs/{job_id}/comments", response_model=CommentResponse)
async def create_comment(
    job_id: str,
    data: CommentCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """发表评论"""
    # 检查作品是否存在且已发布
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="作品不存在")
    if not job.is_public:
        raise HTTPException(status_code=400, detail="只能对已发布的作品评论")
    
    # 创建评论
    comment = Comment(
        user_id=user.id,
        job_id=job_id,
        content=data.content.strip(),
    )
    db.add(comment)
    await db.commit()
    await db.refresh(comment)
    
    return CommentResponse(
        id=comment.id,
        content=comment.content,
        created_at=comment.created_at,
        user={
            "username": user.username,
            "nickname": user.nickname,
            "avatar_url": user.avatar_url,
        }
    )


@router.get("/jobs/{job_id}/comments", response_model=CommentsListResponse)
async def get_comments(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    page: int = 1,
    limit: int = 20,
):
    """获取评论列表"""
    offset = (page - 1) * limit
    
    # 获取评论总数
    count_result = await db.execute(
        select(func.count(Comment.id)).where(Comment.job_id == job_id)
    )
    total = count_result.scalar() or 0
    
    # 获取评论列表
    result = await db.execute(
        select(Comment)
        .where(Comment.job_id == job_id)
        .order_by(Comment.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    comments = result.scalars().all()
    
    # 获取用户信息
    user_ids = [c.user_id for c in comments]
    if user_ids:
        users_result = await db.execute(
            select(User).where(User.id.in_(user_ids))
        )
        users_map = {u.id: u for u in users_result.scalars().all()}
    else:
        users_map = {}
    
    return CommentsListResponse(
        comments=[
            CommentResponse(
                id=c.id,
                content=c.content,
                created_at=c.created_at,
                user={
                    "username": users_map.get(c.user_id, {}).username if c.user_id in users_map else "未知用户",
                    "nickname": users_map.get(c.user_id, {}).nickname if c.user_id in users_map else None,
                    "avatar_url": users_map.get(c.user_id, {}).avatar_url if c.user_id in users_map else None,
                }
            )
            for c in comments
        ],
        total=total,
    )


@router.delete("/comments/{comment_id}")
async def delete_comment(
    comment_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除评论（只能删除自己的）"""
    result = await db.execute(
        select(Comment).where(Comment.id == comment_id)
    )
    comment = result.scalar_one_or_none()
    
    if not comment:
        raise HTTPException(status_code=404, detail="评论不存在")
    
    if comment.user_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="只能删除自己的评论")
    
    await db.delete(comment)
    await db.commit()
    
    return {"success": True, "message": "评论已删除"}


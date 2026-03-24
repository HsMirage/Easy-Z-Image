# -*- coding: utf-8 -*-
"""API 依赖"""
from datetime import datetime
from typing import Optional
from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import jwt, JWTError

from app.models import get_db, User, Worker
from app.config import settings

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """获取当前登录用户"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录",
        )
    
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        sub = payload.get("sub")
        if sub is None:
            raise HTTPException(status_code=401, detail="无效的令牌")
        user_id = int(sub)  # sub 是字符串，转回整数
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="无效的令牌")
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="账号已禁用")
    
    # 重置每日配额
    user.reset_daily_quota_if_needed()
    
    return user


async def get_current_admin(
    user: User = Depends(get_current_user),
) -> User:
    """获取当前管理员用户"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


async def verify_worker_auth(
    x_worker_id: str = Header(...),
    x_api_key: str = Header(...),
    db: AsyncSession = Depends(get_db),
) -> Worker:
    """验证 Worker 认证"""
    if x_api_key != settings.WORKER_API_KEY:
        raise HTTPException(status_code=401, detail="Worker API Key 无效")
    
    # 获取或创建 Worker
    result = await db.execute(select(Worker).where(Worker.id == x_worker_id))
    worker = result.scalar_one_or_none()
    
    if not worker:
        worker = Worker(id=x_worker_id, name=x_worker_id)
        db.add(worker)
        await db.flush()
    
    return worker



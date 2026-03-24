# -*- coding: utf-8 -*-
"""认证 API - Linux DO Connect OAuth"""
from datetime import datetime, timedelta
from urllib.parse import urlencode
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from jose import jwt
import httpx

from app.models import get_db, User
from app.config import settings
from app.api.deps import get_current_user

router = APIRouter()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserInfo(BaseModel):
    id: int
    username: str
    nickname: str | None
    avatar_url: str | None
    is_admin: bool
    trust_level: int
    daily_quota: int
    today_used_count: int
    remaining_quota: int
    total_generations: int


def create_access_token(user_id: int) -> str:
    """创建访问令牌"""
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


@router.get("/login")
async def login_redirect():
    """
    重定向到 Linux DO Connect 授权页面
    """
    if not settings.LINUX_DO_CLIENT_ID:
        raise HTTPException(status_code=500, detail="OAuth 未配置，请联系管理员")
    
    params = {
        "client_id": settings.LINUX_DO_CLIENT_ID,
        "redirect_uri": settings.LINUX_DO_REDIRECT_URI,
        "response_type": "code",
        "scope": "user",
    }
    auth_url = f"{settings.LINUX_DO_AUTH_URL}?{urlencode(params)}"
    
    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def oauth_callback(
    code: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Linux DO Connect OAuth 回调处理
    """
    if not settings.LINUX_DO_CLIENT_ID:
        raise HTTPException(status_code=500, detail="OAuth 未配置")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: 用授权码换取 access_token
            print(f"[OAuth] Requesting token with code: {code[:10]}...")
            token_resp = await client.post(
                settings.LINUX_DO_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": settings.LINUX_DO_CLIENT_ID,
                    "client_secret": settings.LINUX_DO_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": settings.LINUX_DO_REDIRECT_URI,
                },
            )
            print(f"[OAuth] Token response status: {token_resp.status_code}")
            print(f"[OAuth] Token response: {token_resp.text[:500]}")
            
            token_data = token_resp.json()
            
            if "access_token" not in token_data:
                print(f"[OAuth] Token error: {token_data}")
                error_msg = token_data.get("error_description", token_data.get("error", "获取令牌失败"))
                raise HTTPException(status_code=400, detail=f"获取令牌失败: {error_msg}")
            
            linux_do_token = token_data["access_token"]
            
            # Step 2: 获取用户信息
            user_resp = await client.get(
                settings.LINUX_DO_USERINFO_URL,
                headers={"Authorization": f"Bearer {linux_do_token}"},
            )
            user_data = user_resp.json()
            print(f"[OAuth] User data: {user_data}")
            
    except httpx.RequestError as e:
        print(f"[OAuth] Request error: {e}")
        raise HTTPException(status_code=400, detail=f"OAuth 请求失败: {str(e)}")
    except Exception as e:
        print(f"[OAuth] Error: {e}")
        raise HTTPException(status_code=400, detail=f"OAuth 失败: {str(e)}")
    
    # 解析用户信息
    linux_do_id = str(user_data.get("id"))
    username = user_data.get("username", linux_do_id)
    nickname = user_data.get("name")
    trust_level = user_data.get("trust_level", 0)
    is_silenced = user_data.get("silenced", False)
    
    # 处理头像 URL
    avatar_template = user_data.get("avatar_template", "")
    avatar_url = None
    if avatar_template:
        # 替换尺寸占位符，使用 128px
        avatar_url = avatar_template.replace("{size}", "128")
        if not avatar_url.startswith("http"):
            avatar_url = f"https://linux.do{avatar_url}"
    
    # 检查是否被禁言
    if is_silenced:
        raise HTTPException(status_code=403, detail="您的账号已被禁言，无法使用本服务")
    
    # 获取或创建用户
    result = await db.execute(
        select(User).where(User.linux_do_user_id == linux_do_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        # 创建新用户
        user = User(
            linux_do_user_id=linux_do_id,
            username=username,
            nickname=nickname,
            avatar_url=avatar_url,
            trust_level=trust_level,
            is_silenced=is_silenced,
        )
        user.update_quota_by_trust_level()
        db.add(user)
        await db.flush()
        print(f"[OAuth] New user created: {username} (TL{trust_level}, quota={user.daily_quota})")
    else:
        # 更新现有用户信息
        user.username = username
        user.nickname = nickname
        user.avatar_url = avatar_url
        user.trust_level = trust_level
        user.is_silenced = is_silenced
        user.update_quota_by_trust_level()
        print(f"[OAuth] User updated: {username} (TL{trust_level}, quota={user.daily_quota})")
    
    await db.commit()
    
    # 生成本系统的 JWT
    access_token = create_access_token(user.id)
    
    # 重定向回前端，携带 token
    redirect_url = f"{settings.FRONTEND_URL}/auth/callback?token={access_token}"
    return RedirectResponse(url=redirect_url)


@router.post("/dev-login")
async def dev_login(
    username: str = Query(default="dev_user"),
    password: str = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    管理员登录或开发环境快速登录
    管理员账号通过环境变量 ADMIN_USERNAME 和 ADMIN_PASSWORD 配置
    """
    # 管理员登录验证（任何环境都可用）
    is_admin = False
    if settings.ADMIN_USERNAME and username == settings.ADMIN_USERNAME:
        if not settings.ADMIN_PASSWORD or password != settings.ADMIN_PASSWORD:
            raise HTTPException(status_code=401, detail="管理员密码错误")
        is_admin = True
    elif not settings.DEBUG:
        # 非管理员登录仅在开发环境可用
        raise HTTPException(status_code=403, detail="仅开发环境可用")
    
    # 查找或创建用户
    result = await db.execute(
        select(User).where(User.linux_do_user_id == f"dev_{username}")
    )
    user = result.scalar_one_or_none()
    
    if not user:
        user = User(
            linux_do_user_id=f"dev_{username}",
            username=username,
            nickname="管理员" if is_admin else f"开发用户 {username}",
            trust_level=4 if is_admin else 2,
            is_admin=is_admin,
            daily_quota=settings.ADMIN_DAILY_QUOTA if is_admin else settings.DEFAULT_DAILY_QUOTA,
        )
        if not is_admin:
            user.update_quota_by_trust_level()
        db.add(user)
        await db.flush()
    else:
        # 更新管理员状态
        user.is_admin = is_admin
        if is_admin:
            user.daily_quota = settings.ADMIN_DAILY_QUOTA
            user.trust_level = 4
            user.nickname = "管理员"
        else:
            user.update_quota_by_trust_level()
    
    await db.commit()
    
    access_token = create_access_token(user.id)
    
    return TokenResponse(
        access_token=access_token,
        user={
            "id": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "is_admin": user.is_admin,
            "trust_level": user.trust_level,
            "daily_quota": user.daily_quota,
        },
    )


@router.get("/me", response_model=UserInfo)
async def get_me(user: User = Depends(get_current_user)):
    """获取当前用户信息"""
    return UserInfo(
        id=user.id,
        username=user.username,
        nickname=user.nickname,
        avatar_url=user.avatar_url,
        is_admin=user.is_admin,
        trust_level=user.trust_level,
        daily_quota=user.daily_quota,
        today_used_count=user.today_used_count,
        remaining_quota=user.remaining_quota,
        total_generations=user.total_generations,
    )

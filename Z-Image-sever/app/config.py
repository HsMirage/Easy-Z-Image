# -*- coding: utf-8 -*-
"""服务器配置"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""
    
    # 基础配置
    APP_NAME: str = "Z-Image Service"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # 数据库
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/zimage.db")
    
    # JWT 配置
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 天
    
    # Linux DO Connect OAuth
    LINUX_DO_CLIENT_ID: str = os.getenv("LINUX_DO_CLIENT_ID", "")
    LINUX_DO_CLIENT_SECRET: str = os.getenv("LINUX_DO_CLIENT_SECRET", "")
    LINUX_DO_REDIRECT_URI: str = os.getenv("LINUX_DO_REDIRECT_URI", "http://localhost:8000/api/auth/callback")
    LINUX_DO_AUTH_URL: str = "https://connect.linux.do/oauth2/authorize"
    LINUX_DO_TOKEN_URL: str = "https://connect.linux.do/oauth2/token"
    LINUX_DO_USERINFO_URL: str = "https://connect.linux.do/api/user"
    
    # 前端 URL（OAuth 回调后重定向）
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # 管理员账号（开发环境登录）
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")
    
    # Worker 配置
    WORKER_API_KEY: str = os.getenv("WORKER_API_KEY", "dev-api-key-change-in-production")
    WORKER_HEARTBEAT_TIMEOUT: int = 30  # 秒
    
    # 任务配置
    MAX_QUEUE_LENGTH: int = 50  # 队列软上限
    HARD_QUEUE_LIMIT: int = 500  # 硬上限
    JOB_TIMEOUT_SECONDS: int = 300
    MAX_RETRY_COUNT: int = 1
    REFUND_QUOTA_ON_FAILURE: bool = True
    
    # 配额配置（基于 Linux DO trust_level）
    # trust_level 0-1: 1张/天, 2: 5张/天, 3-4: 20张/天
    QUOTA_BY_TRUST_LEVEL: dict = {
        0: 1,   # 新用户
        1: 1,   # 基础用户
        2: 5,   # 成员
        3: 20,  # 常规
        4: 20,  # 领导者
    }
    DEFAULT_DAILY_QUOTA: int = 1  # 默认配额（未登录或无 trust_level）
    ADMIN_DAILY_QUOTA: int = 1000
    
    # 分辨率限制
    MAX_WIDTH: int = 1024
    MAX_HEIGHT: int = 1024
    
    # 存储配置
    STORAGE_ROOT: Path = Path(os.getenv("STORAGE_ROOT", "./storage"))
    
    class Config:
        env_file = ".env"


settings = Settings()

# 确保存储目录存在
settings.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)



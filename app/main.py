# -*- coding: utf-8 -*-
"""
Z-Image 服务端主应用

FastAPI 应用，提供：
- 用户认证 (linux.do OAuth)
- 任务管理
- Worker 协调
- 画廊展示
- 管理后台
"""
import asyncio
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app.config import settings
from app.models import init_db
from app.models.database import async_session
from app.models.job import Job, JobStatus
from app.api import api_router


class CachedStaticFiles(StaticFiles):
    """Serve generated images with a long-lived cache policy."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response


async def cleanup_stale_jobs():
    """定时清理超时任务（仅对 running 状态，从 started_at 计算 5 分钟）"""
    while True:
        try:
            await asyncio.sleep(60)  # 每分钟检查一次
            
            async with async_session() as db:
                timeout_threshold = datetime.utcnow() - timedelta(seconds=settings.JOB_TIMEOUT_SECONDS)
                
                # 只查找 running 状态且 started_at 超时的任务
                # queued 状态的任务不设超时（排队等待是正常的）
                result = await db.execute(
                    select(Job).where(
                        Job.status == JobStatus.RUNNING.value,
                        Job.started_at.isnot(None),
                        Job.started_at < timeout_threshold
                    )
                )
                stale_jobs = result.scalars().all()
                
                for job in stale_jobs:
                    job.status = JobStatus.FAILED.value
                    job.error_message = "生成超时（5分钟），已自动取消"
                    job.finished_at = datetime.utcnow()
                    print(f"[Cleanup] Job {job.id} timed out (running for too long), marked as failed")
                
                if stale_jobs:
                    await db.commit()
                    print(f"[Cleanup] Cleaned up {len(stale_jobs)} stale running jobs")
                    
        except Exception as e:
            print(f"[Cleanup] Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    # 启动时初始化数据库
    print("[Server] Initializing database...")
    await init_db()
    print("[Server] Database initialized")
    
    # 启动后台清理任务
    cleanup_task = asyncio.create_task(cleanup_stale_jobs())
    print("[Server] Started stale job cleanup task")
    
    yield
    
    # 关闭时清理
    cleanup_task.cancel()
    print("[Server] Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Z-Image 在线生图服务 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(api_router, prefix="/api")

# 静态文件服务（存储的图片）
if settings.STORAGE_ROOT.exists():
    app.mount("/storage", CachedStaticFiles(directory=settings.STORAGE_ROOT), name="storage")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )



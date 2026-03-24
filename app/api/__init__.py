# -*- coding: utf-8 -*-
"""API 路由"""
from fastapi import APIRouter

from app.api import admin, auth, chat, gallery, jobs, social, worker_api, workers

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["认证"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["任务"])
api_router.include_router(workers.router, prefix="/workers", tags=["Worker"])
api_router.include_router(worker_api.router, prefix="/worker-api", tags=["通用 Worker API"])
api_router.include_router(gallery.router, prefix="/gallery", tags=["画廊"])
api_router.include_router(admin.router, prefix="/admin", tags=["管理"])
api_router.include_router(social.router, prefix="/social", tags=["社交"])
api_router.include_router(chat.router, prefix="/chat", tags=["聊天室"])




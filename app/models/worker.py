# -*- coding: utf-8 -*-
"""Worker 模型"""
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, String, DateTime, JSON

from app.models.database import Base
from app.config import settings


class WorkerStatus(str, Enum):
    """Worker 状态"""
    IDLE = "idle"
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"


class Worker(Base):
    """Worker 表"""
    __tablename__ = "workers"
    
    id = Column(String(100), primary_key=True, index=True)
    name = Column(String(200))
    status = Column(String(20), default=WorkerStatus.OFFLINE.value)
    
    current_job_id = Column(String(36))
    gpu_info = Column(JSON)
    
    last_seen_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    @property
    def is_online(self) -> bool:
        """检查是否在线"""
        if not self.last_seen_at:
            return False
        elapsed = (datetime.utcnow() - self.last_seen_at).total_seconds()
        return elapsed < settings.WORKER_HEARTBEAT_TIMEOUT




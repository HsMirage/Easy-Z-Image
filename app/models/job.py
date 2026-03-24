# -*- coding: utf-8 -*-
"""任务模型"""
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship

from app.models.database import Base


class JobStatus(str, Enum):
    """任务状态"""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """任务表"""
    __tablename__ = "jobs"
    
    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    worker_id = Column(String(100), index=True)
    
    status = Column(String(20), default=JobStatus.QUEUED.value, index=True)
    priority = Column(Integer, default=0, index=True)  # 优先级，数字越大越优先
    
    # 生成参数
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text)
    width = Column(Integer, default=1024)
    height = Column(Integer, default=1024)
    steps = Column(Integer, default=9)
    cfg_scale = Column(Integer, default=0)  # Turbo 模型无效
    seed = Column(Integer, default=-1)
    sampler = Column(String(50))
    
    # 结果
    image_path = Column(String(500))
    result_metadata = Column(JSON)  # 改名避免与 SQLAlchemy 保留字冲突
    
    # 发布到广场
    is_public = Column(Boolean, default=False, index=True)  # 是否公开到广场
    public_status = Column(String(20), default="private", index=True)  # private/pending/approved/rejected
    is_anonymous = Column(Boolean, default=True)  # 是否匿名显示
    is_nsfw = Column(Boolean, default=False, index=True)  # 是否为 NSFW 内容
    reviewed_at = Column(DateTime)  # 最近一次审核时间
    
    # 软删除
    is_deleted = Column(Boolean, default=False, index=True)  # 用户删除标记
    deleted_at = Column(DateTime)  # 删除时间
    
    # 错误信息
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # 时间
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    
    # 关系
    user = relationship("User", back_populates="jobs")
    
    @property
    def elapsed_seconds(self) -> float | None:
        """耗时（秒）"""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_pending(self) -> bool:
        """是否为待处理状态"""
        return self.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value)


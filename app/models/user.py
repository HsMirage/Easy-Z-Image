# -*- coding: utf-8 -*-
"""用户模型"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Date
from sqlalchemy.orm import relationship

from app.models.database import Base
from app.config import settings


class User(Base):
    """用户表"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    linux_do_user_id = Column(String(100), unique=True, index=True, nullable=False)
    username = Column(String(100), nullable=False)
    nickname = Column(String(200))
    avatar_url = Column(String(500))
    
    # Linux DO 信息
    trust_level = Column(Integer, default=0)  # 信任等级 0-4
    is_silenced = Column(Boolean, default=False)  # 禁言状态
    
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # 配额（默认根据 trust_level 自动计算，可由管理员单独覆盖）
    daily_quota = Column(Integer, default=settings.DEFAULT_DAILY_QUOTA)
    quota_override = Column(Integer, nullable=True)
    today_used_count = Column(Integer, default=0)
    last_reset_date = Column(Date, default=date.today)
    
    # 统计
    total_generations = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    jobs = relationship("Job", back_populates="user")
    likes = relationship("Like", back_populates="user")
    comments = relationship("Comment", back_populates="user")
    chat_messages = relationship("ChatMessage", back_populates="user")
    
    def reset_daily_quota_if_needed(self):
        """如果跨天则重置每日配额"""
        today = date.today()
        if self.last_reset_date != today:
            self.today_used_count = 0
            self.last_reset_date = today
            
    @property
    def remaining_quota(self) -> int:
        """剩余配额"""
        self.reset_daily_quota_if_needed()
        return max(0, self.daily_quota - self.today_used_count)
    
    def can_submit_job(self) -> bool:
        """是否可以提交任务"""
        return self.remaining_quota > 0
    
    def update_quota_by_trust_level(self):
        """根据 trust_level 更新每日配额"""
        if self.is_admin:
            self.daily_quota = settings.ADMIN_DAILY_QUOTA
        elif self.quota_override is not None:
            self.daily_quota = self.quota_override
        else:
            self.daily_quota = settings.QUOTA_BY_TRUST_LEVEL.get(
                self.trust_level, 
                settings.DEFAULT_DAILY_QUOTA
            )


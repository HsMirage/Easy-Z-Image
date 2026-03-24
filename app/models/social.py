# -*- coding: utf-8 -*-
"""社交功能模型：点赞和评论"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.models.database import Base


class Like(Base):
    """点赞表"""
    __tablename__ = "likes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 唯一约束：每个用户对每个作品只能点赞一次
    __table_args__ = (
        UniqueConstraint('user_id', 'job_id', name='unique_user_job_like'),
    )
    
    # 关联
    user = relationship("User", back_populates="likes")


class Comment(Base):
    """评论表"""
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关联用户信息
    user = relationship("User", back_populates="comments")


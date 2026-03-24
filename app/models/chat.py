# -*- coding: utf-8 -*-
"""聊天室模型"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.models.database import Base


class ChatMessage(Base):
    """聊天消息"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # 可为空（匿名用户）
    display_name = Column(String(50), nullable=False)  # 显示名称
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # 关联
    user = relationship("User", back_populates="chat_messages")


class ChatActivityLog(Base):
    """聊天室活动日志（加入/离开）"""
    __tablename__ = "chat_activity_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    username = Column(String(100), nullable=True)  # 真实用户名
    display_name = Column(String(50), nullable=False)  # 显示名称
    activity_type = Column(String(10), nullable=False)  # 'join' or 'leave'
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


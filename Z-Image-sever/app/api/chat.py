# -*- coding: utf-8 -*-
"""聊天室 API"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Set, Optional, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from jose import jwt, JWTError

from app.models.database import get_db
from app.models.user import User
from app.models.chat import ChatMessage, ChatActivityLog
from app.config import settings

router = APIRouter()

# 在线用户管理
class ConnectionManager:
    def __init__(self):
        # websocket -> user_info
        self.active_connections: Dict[WebSocket, dict] = {}
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, user_info: dict):
        await websocket.accept()
        async with self.lock:
            self.active_connections[websocket] = user_info
    
    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                del self.active_connections[websocket]
    
    async def broadcast(self, message: dict, activity: dict = None):
        """广播消息给所有连接，管理员额外收到在线用户列表和活动日志"""
        async with self.lock:
            dead_connections = []
            online_users = self._get_online_users_internal(include_details=True)
            
            for connection, info in self.active_connections.items():
                try:
                    msg = message.copy()
                    # 管理员额外信息
                    if info.get('is_admin'):
                        msg['online_users'] = online_users
                        if activity:
                            msg['activity'] = activity
                    await connection.send_json(msg)
                except:
                    dead_connections.append(connection)
            
            # 清理断开的连接
            for conn in dead_connections:
                if conn in self.active_connections:
                    del self.active_connections[conn]
    
    def get_online_count(self) -> int:
        return len(self.active_connections)
    
    def _get_online_users_internal(self, include_details: bool = False) -> list:
        """内部方法：获取在线用户列表"""
        users = []
        seen = set()
        for ws, info in self.active_connections.items():
            display_name = info.get('display_name', '匿名用户')
            username = info.get('username')
            key = (display_name, username)
            if key not in seen:
                seen.add(key)
                user_data = {'display_name': display_name}
                if include_details:
                    user_data['user_id'] = info.get('user_id')
                    user_data['username'] = username
                users.append(user_data)
        return users
    
    def get_online_users(self, include_user_id: bool = False) -> list:
        """获取在线用户列表"""
        return self._get_online_users_internal(include_details=include_user_id)

manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
    nickname: Optional[str] = Query(default=None),
):
    """WebSocket 聊天端点"""
    # 尝试解析用户身份
    user_id = None
    username = None
    is_admin = False
    avatar_url = None
    
    if token:
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM],
            )
            user_id = int(payload.get("sub")) if payload.get("sub") else None
            # 从数据库获取用户信息
            from app.models.database import async_session
            async with async_session() as db:
                result = await db.execute(select(User).where(User.id == user_id))
                user = result.scalar_one_or_none()
                if user:
                    username = user.username
                    is_admin = user.is_admin
                    avatar_url = user.avatar_url
        except (JWTError, ValueError):
            pass
    
    # 设置显示名称：如果有 nickname 参数则使用匿名昵称，否则使用真实用户名
    # 匿名模式不显示头像
    is_anonymous = bool(nickname and nickname.strip())
    if is_anonymous:
        display_name = nickname.strip()[:20]
        display_avatar = None  # 匿名模式不显示头像
    elif username:
        display_name = username
        display_avatar = avatar_url
    else:
        display_name = f"游客{id(websocket) % 10000:04d}"
        display_avatar = None
    
    user_info = {
        'user_id': user_id,
        'username': username,
        'display_name': display_name,
        'avatar_url': display_avatar,
        'is_admin': is_admin,
    }
    
    await manager.connect(websocket, user_info)
    
    # 保存加入活动日志
    await save_activity_log(user_id, username, display_name, 'join')
    
    # 如果是管理员，先发送历史活动日志
    if is_admin:
        activity_logs = await get_activity_logs(100)
        await websocket.send_json({
            'type': 'init',
            'activity_logs': activity_logs,
            'online_count': manager.get_online_count(),
            'online_users': manager._get_online_users_internal(include_details=True),
        })
    
    # 广播在线人数更新，管理员可见活动日志
    await manager.broadcast(
        {
            'type': 'online_update',
            'online_count': manager.get_online_count(),
            'timestamp': datetime.utcnow().isoformat(),
        },
        activity={
            'type': 'join',
            'display_name': display_name,
            'username': username,
            'timestamp': datetime.utcnow().isoformat(),
        }
    )
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg_data = json.loads(data)
                content = msg_data.get('content', '').strip()
                
                if not content or len(content) > 500:
                    continue
                
                # 先保存消息到数据库获取ID
                msg_id = await save_message(user_id, display_name, content)
                
                # 构建消息（包含ID供删除使用）
                message = {
                    'type': 'message',
                    'id': msg_id,
                    'display_name': display_name,
                    'avatar_url': display_avatar,
                    'content': content,
                    'timestamp': datetime.utcnow().isoformat(),
                    'is_admin': is_admin,
                }
                
                await manager.broadcast(message)
                
            except json.JSONDecodeError:
                continue
                
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        # 保存离开活动日志
        await save_activity_log(user_id, username, display_name, 'leave')
        # 广播在线人数更新，管理员可见活动日志
        await manager.broadcast(
            {
                'type': 'online_update',
                'online_count': manager.get_online_count(),
                'timestamp': datetime.utcnow().isoformat(),
            },
            activity={
                'type': 'leave',
                'display_name': display_name,
                'username': username,
                'timestamp': datetime.utcnow().isoformat(),
            }
        )


async def save_message(user_id: Optional[int], display_name: str, content: str) -> int:
    """保存聊天消息并返回ID"""
    from app.models.database import async_session
    
    async with async_session() as db:
        msg = ChatMessage(
            user_id=user_id,
            display_name=display_name,
            content=content,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        return msg.id


async def save_activity_log(user_id: Optional[int], username: Optional[str], display_name: str, activity_type: str):
    """保存活动日志"""
    from app.models.database import async_session
    
    async with async_session() as db:
        log = ChatActivityLog(
            user_id=user_id,
            username=username,
            display_name=display_name,
            activity_type=activity_type,
        )
        db.add(log)
        await db.commit()


async def get_activity_logs(limit: int = 100) -> list:
    """获取活动日志"""
    from app.models.database import async_session
    
    async with async_session() as db:
        result = await db.execute(
            select(ChatActivityLog)
            .order_by(desc(ChatActivityLog.created_at))
            .limit(limit)
        )
        logs = result.scalars().all()
        
        return [
            {
                'type': log.activity_type,
                'display_name': log.display_name,
                'username': log.username,
                'timestamp': log.created_at.isoformat(),
            }
            for log in reversed(logs)
        ]


@router.get("/messages")
async def get_messages(
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """获取历史消息"""
    result = await db.execute(
        select(ChatMessage)
        .order_by(desc(ChatMessage.created_at))
        .limit(limit)
    )
    messages = result.scalars().all()
    
    return {
        'messages': [
            {
                'id': msg.id,
                'display_name': msg.display_name,
                'content': msg.content,
                'timestamp': msg.created_at.isoformat(),
            }
            for msg in reversed(messages)
        ]
    }


@router.get("/messages/admin")
async def get_messages_admin(
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """管理员获取历史消息（包含用户ID）"""
    result = await db.execute(
        select(ChatMessage)
        .order_by(desc(ChatMessage.created_at))
        .limit(limit)
    )
    messages = result.scalars().all()
    
    return {
        'messages': [
            {
                'id': msg.id,
                'user_id': msg.user_id,
                'display_name': msg.display_name,
                'content': msg.content,
                'timestamp': msg.created_at.isoformat(),
            }
            for msg in reversed(messages)
        ]
    }


@router.get("/online")
async def get_online_info():
    """获取在线信息"""
    return {
        'online_count': manager.get_online_count(),
        'users': manager.get_online_users(),
    }


@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: int,
    db: AsyncSession = Depends(get_db),
):
    """管理员删除聊天消息"""
    # 查找消息
    result = await db.execute(select(ChatMessage).where(ChatMessage.id == message_id))
    message = result.scalar_one_or_none()
    
    if not message:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="消息不存在")
    
    await db.delete(message)
    await db.commit()
    
    # 广播删除消息事件
    await manager.broadcast({
        'type': 'delete',
        'message_id': message_id,
        'timestamp': datetime.utcnow().isoformat(),
    })
    
    return {"success": True}

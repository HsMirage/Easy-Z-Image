# -*- coding: utf-8 -*-
"""数据库配置"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import inspect, text

from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    """获取数据库会话"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """初始化数据库"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(run_migrations)


def run_migrations(sync_conn):
    """执行轻量级数据库迁移"""
    inspector = inspect(sync_conn)
    if "users" not in inspector.get_table_names():
        user_columns = set()
    else:
        user_columns = {column["name"] for column in inspector.get_columns("users")}

    if "quota_override" not in user_columns:
        sync_conn.execute(text("ALTER TABLE users ADD COLUMN quota_override INTEGER"))

    if "jobs" not in inspector.get_table_names():
        return

    job_columns = {column["name"] for column in inspector.get_columns("jobs")}

    if "public_status" not in job_columns:
        sync_conn.execute(text("ALTER TABLE jobs ADD COLUMN public_status VARCHAR(20) DEFAULT 'private'"))
        sync_conn.execute(
            text(
                "UPDATE jobs "
                "SET public_status = CASE WHEN is_public = 1 THEN 'approved' ELSE 'private' END"
            )
        )
    else:
        sync_conn.execute(
            text(
                "UPDATE jobs "
                "SET public_status = CASE "
                "WHEN is_public = 1 THEN 'approved' "
                "ELSE COALESCE(NULLIF(public_status, ''), 'private') "
                "END "
                "WHERE public_status IS NULL OR public_status = ''"
            )
        )

    if "is_nsfw" not in job_columns:
        sync_conn.execute(text("ALTER TABLE jobs ADD COLUMN is_nsfw BOOLEAN DEFAULT 0"))

    if "reviewed_at" not in job_columns:
        sync_conn.execute(text("ALTER TABLE jobs ADD COLUMN reviewed_at DATETIME"))




# -*- coding: utf-8 -*-
"""Worker 配置"""
import os
from pathlib import Path

# 加载 .env 文件 (从 worker 目录)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv 未安装时使用系统环境变量

# Worker 标识
WORKER_ID = os.getenv("WORKER_ID", "worker-default")
WORKER_NAME = os.getenv("WORKER_NAME", "Z-Image Worker")

# 远程后端配置
REMOTE_API_BASE = os.getenv("REMOTE_API_BASE", "http://localhost:8000")
API_KEY = os.getenv("WORKER_API_KEY", "dev-api-key-change-in-production")

# 本地 Z-Image 配置
MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
_rev = os.getenv("MODEL_REVISION", "")
MODEL_REVISION = _rev.strip() if _rev else None  # None = 使用最新版本
DEVICE = os.getenv("DEVICE", "cuda")
USE_CPU_OFFLOAD = os.getenv("USE_CPU_OFFLOAD", "true").lower() == "true"

# 时间配置（秒）
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "2"))
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "300"))

# 本地备份路径（默认为 Worker 目录下的 storage）
LOCAL_BACKUP_ROOT = Path(os.getenv("LOCAL_BACKUP_ROOT", str(Path(__file__).parent / "storage")))

# 生成参数默认值
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 9  # Z-Image-Turbo 推荐 8-9 步
MAX_WIDTH = 1024
MAX_HEIGHT = 1024

# GPU 信息（用于心跳上报，会被实际检测值覆盖）
GPU_INFO = {
    "name": "NVIDIA GPU",
    "memory_gb": 8,
}




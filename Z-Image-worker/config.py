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
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "auto").strip().lower()
MULTI_GPU_MODE = os.getenv("MULTI_GPU_MODE", "auto").strip().lower()
MULTI_GPU_DEVICES = os.getenv("MULTI_GPU_DEVICES", "").strip()
GPU_MEMORY_RESERVE_GB = float(os.getenv("GPU_MEMORY_RESERVE_GB", "0.5"))
MULTI_GPU_VAE_DECODE_RESERVE_GB = float(os.getenv("MULTI_GPU_VAE_DECODE_RESERVE_GB", "0.5"))
MULTI_GPU_ACTIVATION_RESERVE_GB = float(os.getenv("MULTI_GPU_ACTIVATION_RESERVE_GB", "1.0"))
USE_CPU_OFFLOAD = os.getenv("USE_CPU_OFFLOAD", "true").lower() == "true"
_multi_gpu_enabled_by_config = (
    DEVICE == "cuda" and MULTI_GPU_MODE not in {"off", "disabled", "false", "0"}
)
_unload_default = "false" if _multi_gpu_enabled_by_config and not USE_CPU_OFFLOAD else "true"
WORKER_UNLOAD_MODEL_AFTER_JOB = os.getenv("UNLOAD_MODEL_AFTER_JOB", _unload_default).lower() == "true"
WORKER_UNLOAD_MODEL_ON_ERROR = os.getenv("UNLOAD_MODEL_ON_ERROR", "true").lower() == "true"
_preload_default = "true" if _multi_gpu_enabled_by_config and not WORKER_UNLOAD_MODEL_AFTER_JOB else "false"
PRELOAD_MODEL_ON_START = os.getenv("PRELOAD_MODEL_ON_START", _preload_default).lower() == "true"
WORKER_ENABLE_OPENAI_COMPAT_API = os.getenv("WORKER_ENABLE_OPENAI_COMPAT_API", "true").lower() == "true"

# 兼容旧代码导入，默认仍指向 Worker 侧行为。
UNLOAD_MODEL_AFTER_JOB = WORKER_UNLOAD_MODEL_AFTER_JOB
UNLOAD_MODEL_ON_ERROR = WORKER_UNLOAD_MODEL_ON_ERROR

# 时间配置（秒）
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "2"))
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "300"))

# 本地备份路径（默认为 Worker 目录下的 storage）
LOCAL_BACKUP_ROOT = Path(os.getenv("LOCAL_BACKUP_ROOT", str(Path(__file__).parent / "storage")))

# OpenAI 兼容图片 API 服务配置
OPENAI_COMPAT_HOST = os.getenv("OPENAI_COMPAT_HOST", "0.0.0.0").strip() or "0.0.0.0"
OPENAI_COMPAT_PORT = int(os.getenv("OPENAI_COMPAT_PORT", "8787"))
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "").strip()
OPENAI_COMPAT_MODEL_NAME = os.getenv("OPENAI_COMPAT_MODEL_NAME", MODEL_ID).strip() or MODEL_ID
OPENAI_COMPAT_PUBLIC_BASE_URL = os.getenv("OPENAI_COMPAT_PUBLIC_BASE_URL", "").strip().rstrip("/")
OPENAI_COMPAT_DEFAULT_RESPONSE_FORMAT = (
    os.getenv("OPENAI_COMPAT_DEFAULT_RESPONSE_FORMAT", "url").strip().lower() or "url"
)
OPENAI_COMPAT_MAX_IMAGES_PER_REQUEST = int(os.getenv("OPENAI_COMPAT_MAX_IMAGES_PER_REQUEST", "1"))
OPENAI_COMPAT_SAVE_ROOT = Path(
    os.getenv("OPENAI_COMPAT_SAVE_ROOT", str(LOCAL_BACKUP_ROOT / "openai-compat"))
)

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

# Easy-Z-Image

---

<p align="center"><strong>⭐ <a href="https://github.com/HsMirage/AIPlanHub">AIPlanHub</a> — 一站式对比国内主流 AI 编程订阅方案，覆盖 9 大平台、25+ 套餐，帮你找到性价比最高的选择 ⭐</strong></p>

---

[English](README.en.md) | 中文

基于 [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 的在线 AI 生图平台，采用 Server + Worker 分布式架构。

> 本项目基于 [Linux DO](https://linux.do) 社区大佬 **RyanVan** 开源的 [RyanVan Z-Image](https://linux.do) 二次开发，在原版基础上新增：
> - 多显卡生成支持（GPU 分片，无需 NVLink）
> - OpenAI 兼容 API（`/v1/images/generations` 和 `/v1/chat/completions`）
> - 作品广场分享默认需要管理员审核
> - NSFW 分区（图片默认隐藏，需主动查看）

## 架构概览

```
┌─────────────┐      ┌─────────────────┐      ┌───────────────┐
│   浏览器     │◄────►│   Server (API)  │◄────►│  Worker (GPU) │
│  (前端页面)  │ HTTP │   FastAPI        │  轮询 │  PyTorch      │
└─────────────┘      └────────┬────────┘      └───────────────┘
                              │                        │
                       ┌──────┴──────┐          ┌──────┴──────┐
                       │   SQLite    │          │  Z-Image    │
                       │   Database  │          │  Turbo 模型  │
                       └─────────────┘          └─────────────┘
```

**Server** — FastAPI 后端，负责用户认证、任务队列、画廊展示、社交互动和管理后台。

**Worker** — GPU 节点，拉取任务并调用 Z-Image-Turbo 模型生成图片，同时提供 OpenAI 兼容的本地图片 API。

## 功能特性

- Linux DO Connect OAuth 登录，基于信任等级的配额系统
- 文生图任务队列，支持多 Worker 并行处理
- 作品广场（画廊），支持发布、审核、点赞、评论
- 管理后台：用户管理、任务管理、Worker 监控
- 聊天室
- Worker 端提供 OpenAI 兼容 API（`/v1/images/generations` 和 `/v1/chat/completions`）
- 多 GPU 分片支持（无需 NVLink）
- 显存不足时自动降级到 CPU Offload，单显卡 4GB 显存 + 12GB 内存即可运行
- 支持 Windows（StartWorker.bat）和 Linux（systemd）部署

## 项目结构

```
Easy-Z-Image/
├── Z-Image-sever/           # 服务端
│   ├── app/
│   │   ├── api/             # API 路由
│   │   │   ├── auth.py      #   认证 (Linux DO OAuth)
│   │   │   ├── jobs.py      #   任务管理
│   │   │   ├── gallery.py   #   画廊
│   │   │   ├── social.py    #   点赞/评论
│   │   │   ├── chat.py      #   聊天室
│   │   │   ├── admin.py     #   管理后台
│   │   │   ├── workers.py   #   Worker 状态
│   │   │   └── worker_api.py#   Worker 通信接口
│   │   ├── models/          # 数据模型
│   │   ├── services/        # 业务服务
│   │   ├── config.py        # 配置
│   │   └── main.py          # 应用入口
│   ├── .env.example         # 环境变量模板
│   ├── requirements.txt     # Python 依赖
│   └── run.py               # 启动脚本
│
└── Z-Image-worker/          # Worker 端 (GPU 节点)
    ├── worker.py             # 主程序（任务轮询 + 心跳）
    ├── generator.py          # 模型加载与图片生成
    ├── openai_compat_server.py # OpenAI 兼容 API 服务
    ├── api_client.py         # 与 Server 通信
    ├── config.py             # 配置加载
    ├── download_model.py     # 模型下载
    ├── requirements.txt      # Python 依赖
    ├── StartWorker.bat       # Windows 管理脚本
    └── zimage-worker.service # Linux systemd 服务文件
```

## 快速开始

### Server 部署

**要求：** Python 3.10+

```bash
cd Z-Image-sever

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填写 Linux DO OAuth 凭据等

# 启动服务
python run.py
```

Server 默认运行在 `http://localhost:8000`，API 文档访问 `http://localhost:8000/docs`。

### Worker 部署

**要求：** Python 3.11、NVIDIA GPU、CUDA 11.8+

| 运行模式 | 显存要求 |
|----------|----------|
| CPU Offload（慢） | 4GB 显存 + 12GB 内存 |
| 单显卡（推荐） | 16GB 显存 |
| 多显卡分片 | 多卡显存总和 ≥ 16GB |

#### Windows

```cmd
cd Z-Image-worker
双击 StartWorker.bat
  [6] 安装依赖
  [5] 配置 Worker
  [7] 下载模型 (~25GB)
  [1] 启动 Worker
```

#### Linux

```bash
cd Z-Image-worker

python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env

python download_model.py
python worker.py
```

详细部署说明参见 [Z-Image-worker/SETUP.md](Z-Image-worker/SETUP.md)。

## 配置说明

### Server 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SECRET_KEY` | JWT 签名密钥 | `dev-secret-key-...` |
| `LINUX_DO_CLIENT_ID` | OAuth Client ID | — |
| `LINUX_DO_CLIENT_SECRET` | OAuth Client Secret | — |
| `LINUX_DO_REDIRECT_URI` | OAuth 回调地址 | `http://localhost:8000/api/auth/callback` |
| `FRONTEND_URL` | 前端地址 | `http://localhost:3000` |
| `WORKER_API_KEY` | Worker 认证密钥 | `dev-api-key-...` |
| `ADMIN_USERNAME` | 管理员用户名 | `admin` |
| `ADMIN_PASSWORD` | 管理员密码 | — |

### Worker 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `WORKER_ID` | Worker 唯一标识 | `worker-default` |
| `REMOTE_API_BASE` | Server 地址 | `http://localhost:8000` |
| `WORKER_API_KEY` | 认证密钥（与 Server 一致） | `dev-api-key-...` |
| `MODEL_ID` | 模型 ID | `Tongyi-MAI/Z-Image-Turbo` |
| `MULTI_GPU_MODE` | 多 GPU 模式 (`auto`/`force`/`off`) | `auto` |
| `OPENAI_COMPAT_PORT` | 本地 OpenAI 兼容 API 端口 | `8787` |

## 配额系统

基于 Linux DO 信任等级自动分配每日生成配额：

| 信任等级 | 每日配额 |
|----------|----------|
| 0-1 (新用户) | 1 张 |
| 2 (成员) | 5 张 |
| 3-4 (常规/领导者) | 20 张 |
| 管理员 | 1000 张 |

管理员可为单个用户自定义配额。

## 开源协议

MIT License

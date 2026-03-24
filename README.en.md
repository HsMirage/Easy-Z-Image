# Easy-Z-Image

English | [中文](README.md)

An online AI image generation platform powered by [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), built with a distributed Server + Worker architecture.

## Architecture

```
┌─────────────┐      ┌─────────────────┐      ┌───────────────┐
│   Browser   │◄────►│   Server (API)  │◄────►│  Worker (GPU) │
│  (Frontend) │ HTTP │   FastAPI        │ Poll │  PyTorch      │
└─────────────┘      └────────┬────────┘      └───────────────┘
                              │                        │
                       ┌──────┴──────┐          ┌──────┴──────┐
                       │   SQLite    │          │  Z-Image    │
                       │   Database  │          │  Turbo Model│
                       └─────────────┘          └─────────────┘
```

**Server** — FastAPI backend handling user authentication, job queue, gallery, social features, and admin dashboard.

**Worker** — GPU node that pulls jobs from the queue and generates images using Z-Image-Turbo. Also exposes an OpenAI-compatible local image API.

## Features

- Linux DO Connect OAuth login with trust-level-based quota system
- Text-to-image job queue with multi-Worker parallel processing
- Public gallery with publish review, likes, and comments
- Admin dashboard: user management, job management, Worker monitoring
- Chat room
- OpenAI-compatible API on Worker (`/v1/images/generations`)
- Multi-GPU sharding support (no NVLink required)
- Deployable on Windows (StartWorker.bat) and Linux (systemd)

## Project Structure

```
Easy-Z-Image/
├── Z-Image-sever/           # Server
│   ├── app/
│   │   ├── api/             # API routes
│   │   │   ├── auth.py      #   Authentication (Linux DO OAuth)
│   │   │   ├── jobs.py      #   Job management
│   │   │   ├── gallery.py   #   Gallery
│   │   │   ├── social.py    #   Likes / Comments
│   │   │   ├── chat.py      #   Chat room
│   │   │   ├── admin.py     #   Admin panel
│   │   │   ├── workers.py   #   Worker status
│   │   │   └── worker_api.py#   Worker communication
│   │   ├── models/          # Data models
│   │   ├── services/        # Business services
│   │   ├── config.py        # Configuration
│   │   └── main.py          # App entry point
│   ├── .env.example         # Environment variable template
│   ├── requirements.txt     # Python dependencies
│   └── run.py               # Startup script
│
└── Z-Image-worker/          # Worker (GPU node)
    ├── worker.py             # Main loop (polling + heartbeat)
    ├── generator.py          # Model loading & image generation
    ├── openai_compat_server.py # OpenAI-compatible API server
    ├── api_client.py         # Server communication
    ├── config.py             # Configuration loader
    ├── download_model.py     # Model downloader
    ├── requirements.txt      # Python dependencies
    ├── StartWorker.bat       # Windows management script
    └── zimage-worker.service # Linux systemd service file
```

## Quick Start

### Server

**Requirements:** Python 3.10+

```bash
cd Z-Image-sever

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your Linux DO OAuth credentials, etc.

# Start server
python run.py
```

The server runs at `http://localhost:8000` by default. API docs are available at `http://localhost:8000/docs`.

### Worker

**Requirements:** Python 3.11, NVIDIA GPU (8 GB+ VRAM), CUDA 11.8+

#### Windows

```cmd
cd Z-Image-worker
Double-click StartWorker.bat
  [6] Install dependencies
  [5] Configure Worker
  [7] Download model (~25 GB)
  [1] Start Worker
```

#### Linux

```bash
cd Z-Image-worker

python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cp .env.example .env
# Edit .env

python download_model.py
python worker.py
```

See [Z-Image-worker/SETUP.md](Z-Image-worker/SETUP.md) for detailed deployment instructions.

## Configuration

### Server Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing key | `dev-secret-key-...` |
| `LINUX_DO_CLIENT_ID` | OAuth Client ID | — |
| `LINUX_DO_CLIENT_SECRET` | OAuth Client Secret | — |
| `LINUX_DO_REDIRECT_URI` | OAuth callback URL | `http://localhost:8000/api/auth/callback` |
| `FRONTEND_URL` | Frontend URL | `http://localhost:3000` |
| `WORKER_API_KEY` | Worker auth key | `dev-api-key-...` |
| `ADMIN_USERNAME` | Admin username | `admin` |
| `ADMIN_PASSWORD` | Admin password | — |

### Worker Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKER_ID` | Unique Worker identifier | `worker-default` |
| `REMOTE_API_BASE` | Server URL | `http://localhost:8000` |
| `WORKER_API_KEY` | Auth key (must match Server) | `dev-api-key-...` |
| `MODEL_ID` | Model identifier | `Tongyi-MAI/Z-Image-Turbo` |
| `MULTI_GPU_MODE` | Multi-GPU mode (`auto`/`force`/`off`) | `auto` |
| `OPENAI_COMPAT_PORT` | Local OpenAI-compatible API port | `8787` |

## Quota System

Daily generation quotas are automatically assigned based on Linux DO trust level:

| Trust Level | Daily Quota |
|-------------|-------------|
| 0-1 (New) | 1 image |
| 2 (Member) | 5 images |
| 3-4 (Regular / Leader) | 20 images |
| Admin | 1000 images |

Admins can set custom quotas for individual users.

## License

MIT License

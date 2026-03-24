# Z-Image Worker

A GPU worker for [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) that:

- Polls a remote task queue and generates images
- Exposes a local **OpenAI-compatible image API** (`POST /v1/images/generations`)
- Supports **multi-GPU sharding** (single-process, no NVLink required)
- Runs as a systemd service on Linux or via `StartWorker.bat` on Windows

## Quick Start

### Windows

1. Install Python 3.11 and CUDA 11.8+
2. Copy `.env.example` to `.env` and fill in your settings
3. Double-click `StartWorker.bat`
   - `[6]` Install dependencies
   - `[7]` Download model (~25 GB)
   - `[1]` Start worker

See [SETUP.md](SETUP.md) for detailed instructions.

### Linux (systemd)

```bash
# 1. Create venv and install deps
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
$EDITOR .env

# 3. Download model
python download_model.py

# 4. Install and start service
mkdir -p ~/.config/systemd/user
cp zimage-worker.service ~/.config/systemd/user/
# Edit WorkingDirectory and ExecStart paths in the service file
systemctl --user daemon-reload
systemctl --user enable --now zimage-worker
```

## OpenAI-Compatible API

Once running, the worker exposes:

```
http://localhost:8787/v1
```

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /v1/models` | List available models |
| `POST /v1/images/generations` | Generate image |

Compatible with Cherry Studio and other OpenAI-compatible clients.

## Configuration

All settings are in `.env`. Key options:

| Variable | Default | Description |
|---|---|---|
| `WORKER_ID` | `worker-default` | Unique worker identifier |
| `REMOTE_API_BASE` | — | Remote task queue URL (optional) |
| `WORKER_API_KEY` | — | Auth key for remote queue |
| `MULTI_GPU_MODE` | `auto` | `auto` / `force` / `off` |
| `OPENAI_COMPAT_PORT` | `8787` | Local API port |

See `.env.example` for the full list.

## Requirements

- Python 3.11
- NVIDIA GPU with 8 GB+ VRAM (10 GB+ recommended)
- CUDA 11.8 or 12.1
- 16 GB RAM, 50 GB disk

## File Structure

```
zimage-worker/
├── worker.py                # Main worker loop
├── generator.py             # Model loading & image generation
├── openai_compat_server.py  # OpenAI-compatible HTTP server
├── api_client.py            # Remote task queue client
├── config.py                # Configuration loader
├── download_model.py        # One-shot model downloader
├── StartWorker.bat          # Windows management script
├── zimage-worker.service    # Linux systemd unit
├── .env.example             # Configuration template
└── requirements.txt         # Python dependencies
```

# Z-Image Worker 快速部署

## 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10 | Windows 11 |
| GPU | 8GB 显存 | 10GB+ 显存，可自动启用单进程多卡分片 |
| 内存 | 16 GB | 32 GB |
| 硬盘 | 50 GB 空闲 | SSD 100GB+ |
| CUDA | 11.8+ | 12.1+ |
| Python | **3.11**（必须） | 3.11.x |

## 快速开始

### 0. 开启 Windows 开发者模式（重要！）

**必须先完成此步骤**，否则模型下载会失败！

1. 打开 **设置** → **隐私和安全性** → **开发者选项**
2. 开启 **开发人员模式**
3. 重启电脑

> 这是因为 HuggingFace 缓存系统需要使用符号链接（symlinks）

### 1. 安装 CUDA

从 [NVIDIA 官网](https://developer.nvidia.com/cuda-downloads) 下载安装 CUDA Toolkit。

验证安装：
```cmd
nvidia-smi
```

### 2. 安装 Python 3.11

⚠️ **必须是 Python 3.11！** 3.10/3.12/3.13 均不支持 PyTorch。

**方法一：命令行安装（推荐）**
```cmd
winget install Python.Python.3.11 --source winget
```

**方法二：官网下载**

从 [Python 3.11.9](https://www.python.org/downloads/release/python-3119/) 下载安装。

**安装时务必勾选 "Add Python to PATH"**

验证安装：
```cmd
py -3.11 --version
```

### 3. 运行管理器

> API_KEY 已内置，无需手动配置

双击 `StartWorker.bat`，按照菜单操作：

1. 选择 `[6] 安装/更新依赖` - 自动安装所有依赖（约 5-10 分钟）
2. 选择 `[5] 配置 Worker` - 设置 Worker ID 和名称
3. 选择 `[7] 下载/更新模型` - 下载 AI 模型（约 25GB，首次需要）
4. 选择 `[1] 启动 Worker` - 连接远程任务队列开始运行，同时暴露本地 OpenAI 兼容图片服务

### 4. 验证运行

启动后应看到类似输出：

```
============================================================
  Z-Image Worker
  ID: worker-xxx
  Name: xxx
============================================================

[Worker] Pre-loading model...
[Generator] Model loaded successfully!
[Worker] Started! Polling interval: 2s
```

Worker 启动后，默认会同时提供 OpenAI 兼容图片 API：

```text
http://localhost:8787/v1
```

支持的核心接口：

```text
GET  /v1/models
GET  /health
POST /v1/images/generations
```

这适合给 Cherry Studio 等支持 OpenAI 兼容服务商的工具直接调用。

## 常见问题

### Q: PyTorch 安装失败

**原因**：Python 版本不对

**解决**：必须使用 Python 3.11
```cmd
winget install Python.Python.3.11 --source winget
```

### Q: 模型下载很慢

设置 HuggingFace 镜像：
```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

### Q: 模型下载失败/显示 .incomplete

**原因**：未开启 Windows 开发者模式

**解决**：设置 → 隐私和安全性 → 开发者选项 → 开启，然后重启电脑

### Q: CUDA out of memory

显存不足时会自动启用 CPU Offload，速度会变慢但可以正常运行。
如果机器有多张卡，Worker 会先尝试单进程多卡分片，再回退到 CPU Offload。

### Q: 如何手动指定多卡分片使用哪些 GPU

在 `.env` 中设置：

```env
TORCH_DTYPE=bf16
MULTI_GPU_MODE=auto
MULTI_GPU_DEVICES=0,1
GPU_MEMORY_RESERVE_GB=0.5
MULTI_GPU_VAE_DECODE_RESERVE_GB=0.5
```

- `TORCH_DTYPE=bf16`：Ampere/30 系、40 系 RTX 推荐，Z-Image 在多卡上通常比 fp16 稳定
- `MULTI_GPU_MODE=auto`：默认优先尝试单进程多卡分片
- `MULTI_GPU_MODE=force`：多卡分片失败时直接报错，不回退到单卡/CPU Offload
- `MULTI_GPU_DEVICES=0,1`：只让 0 和 1 号卡参与分片
- `GPU_MEMORY_RESERVE_GB=0.5`：每张卡额外预留的显存，避免装载阶段顶满
- `MULTI_GPU_VAE_DECODE_RESERVE_GB=0.5`：给 VAE 解码预留显存，减少解码时把模块临时挪到 CPU

### Q: 连接服务器失败

检查：
1. 网络是否正常
2. API_KEY 是否正确
3. 服务器地址是否正确

### Q: 如何接入 Cherry Studio

1. 启动 `StartWorker.bat`
2. 选择 `[1] 启动 Worker`
3. 在 Cherry Studio 中新建自定义 OpenAI 服务商
4. Base URL 填：
   `http://localhost:8787/v1`
5. 模型名填：
   `Tongyi-MAI/Z-Image-Turbo`
6. 如果你在 `.env` 里设置了 `OPENAI_COMPAT_API_KEY`，则在 Cherry Studio 里填同一个 Key

可选 `.env` 配置：

```env
OPENAI_COMPAT_HOST=0.0.0.0
OPENAI_COMPAT_PORT=8787
OPENAI_COMPAT_API_KEY=
OPENAI_COMPAT_MODEL_NAME=Tongyi-MAI/Z-Image-Turbo
OPENAI_COMPAT_PUBLIC_BASE_URL=
OPENAI_COMPAT_DEFAULT_RESPONSE_FORMAT=url
OPENAI_COMPAT_MAX_IMAGES_PER_REQUEST=1
WORKER_ENABLE_OPENAI_COMPAT_API=true
```

### Q: Worker 和 OpenAI API 能否同时运行

可以，但推荐方式已经改成“只启动 Worker”。
Worker 会在同一进程内顺带启动 OpenAI 兼容图片 API，和轮询任务共用同一套模型/显存。
如果 Worker 已经在运行，就不要再单独启动 `openai_compat_server.py`。

## 文件结构

```
Z-Image-Worker/
├── StartWorker.bat     # 管理器（双击运行）
├── download_model.py   # 模型下载脚本
├── worker.py           # 主程序
├── openai_compat_server.py  # OpenAI 兼容图片 API 实现（由 Worker 内嵌调用）
├── config.py           # 配置加载
├── generator.py        # 图像生成
├── api_client.py       # 服务器通信
├── .env                # 环境变量（首次配置后生成）
├── venv/               # Python 虚拟环境（安装依赖后生成）
└── storage/            # 本地备份
```

## 获取帮助

如有问题，请联系服务器管理员。

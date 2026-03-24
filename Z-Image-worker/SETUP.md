# Z-Image Worker 快速部署

## 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10 | Windows 11 |
| GPU | 8GB 显存 | 10GB+ 显存 |
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
4. 选择 `[1] 启动 Worker` - 开始运行

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

### Q: 连接服务器失败

检查：
1. 网络是否正常
2. API_KEY 是否正确
3. 服务器地址是否正确

## 文件结构

```
Z-Image-Worker/
├── StartWorker.bat     # 管理器（双击运行）
├── download_model.py   # 模型下载脚本
├── worker.py           # 主程序
├── config.py           # 配置加载
├── generator.py        # 图像生成
├── api_client.py       # 服务器通信
├── .env                # 环境变量（首次配置后生成）
├── venv/               # Python 虚拟环境（安装依赖后生成）
└── storage/            # 本地备份
```

## 获取帮助

如有问题，请联系服务器管理员。

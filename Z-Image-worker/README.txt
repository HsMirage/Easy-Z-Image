═══════════════════════════════════════════════════════════════
                    Z-Image Worker 分发包
═══════════════════════════════════════════════════════════════

这是 Z-Image AI 生图系统的 Worker 端，运行在有 NVIDIA GPU 的电脑上。


【系统要求】

  - Windows 10/11
  - NVIDIA GPU（8GB 显存以上）
  - 内存 16GB 以上
  - 硬盘 50GB 空闲空间
  - CUDA 11.8 或更高版本
  - Python 3.11（必须！3.10/3.12/3.13 不支持）


【部署步骤】

  1. 开启 Windows 开发者模式（重要！）
     设置 → 隐私和安全性 → 开发者选项 → 开启开发人员模式
     然后重启电脑

  2. 安装 CUDA
     从 https://developer.nvidia.com/cuda-downloads 下载安装

  3. 安装 Python 3.11（必须是 3.11 版本）
     方法一：命令行安装（推荐）
       winget install Python.Python.3.11 --source winget
     
     方法二：官网下载
       https://www.python.org/downloads/release/python-3119/
       安装时勾选 "Add Python to PATH"
     
     ⚠️ 注意：Python 3.13/3.12/3.10 均不支持 PyTorch！

  4. 双击运行 StartWorker.bat
     - 选择 [6] 安装依赖（约 5-10 分钟）
     - 选择 [5] 配置 Worker ID 和名称
     - 选择 [7] 下载模型（约 25GB，首次需要）
     - 选择 [1] 启动 Worker

     注：API_KEY 已内置，无需手动配置


【文件说明】

  StartWorker.bat    - 管理器，双击运行
  worker.py          - 主程序
  config.py          - 配置加载
  generator.py       - 图像生成
  api_client.py      - 服务器通信
  download_model.py  - 模型下载脚本
  SETUP.md           - 详细部署文档


【常见问题】

  Q: 模型下载失败/显示 .incomplete
  A: 请确保已开启 Windows 开发者模式并重启

  Q: PyTorch 安装失败
  A: 必须使用 Python 3.11！运行以下命令安装：
     winget install Python.Python.3.11 --source winget

  Q: 生成速度很慢
  A: 8GB 显存会启用 CPU Offload，速度约 2-3 分钟/张
     10GB+ 显存速度约 20-40 秒/张

  Q: 连接服务器失败
  A: 检查 API_KEY 是否正确，网络是否正常


【技术支持】

  如有问题请联系服务器管理员

═══════════════════════════════════════════════════════════════

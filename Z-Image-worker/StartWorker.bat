@echo off
chcp 65001 >nul
title Z-Image Worker Manager

REM ============================================
REM Configuration - Please modify before first use
REM ============================================

REM Server URL (set your own server, or leave empty to use local OpenAI API only)
set "API_BASE=https://your-server.example.com"

REM Worker API Key (get from your server admin)
set "API_KEY=your-api-key-here"

REM ============================================
REM Auto-detect paths (usually no need to modify)
REM ============================================

REM Worker directory (current directory)
set "WORKER_DIR=%~dp0"
set "WORKER_DIR=%WORKER_DIR:~0,-1%"

REM Local backup directory
set "BACKUP_DIR=%WORKER_DIR%\storage"

REM Auto-detect Python virtual environment
if exist "%WORKER_DIR%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%WORKER_DIR%\venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python.exe"
)

:menu
cls
echo.
echo ============================================================
echo           Z-Image Worker Manager v2.0
echo ============================================================
echo.
echo   [1] Start Worker
echo   [2] Stop Worker / API
echo   [3] Check Status
echo   [4] GPU Info
echo   [5] Configure Worker
echo   [6] Install Dependencies
echo   [7] Download Model
echo   [0] Exit
echo.
echo ============================================================
echo   OpenAI API is exposed automatically by Worker: http://localhost:8787/v1
echo.

REM Show current config
if exist "%WORKER_DIR%\.env" (
    for /f "tokens=2 delims==" %%a in ('findstr "WORKER_ID" "%WORKER_DIR%\.env" 2^>nul') do set "CURRENT_ID=%%a"
    for /f "tokens=2 delims==" %%a in ('findstr "WORKER_NAME" "%WORKER_DIR%\.env" 2^>nul') do set "CURRENT_NAME=%%a"
    echo   Current Worker: %CURRENT_NAME% [%CURRENT_ID%]
    echo.
)

set /p choice=Select [0-7]: 

if "%choice%"=="1" goto start_worker
if "%choice%"=="2" goto stop_worker
if "%choice%"=="3" goto check_status
if "%choice%"=="4" goto gpu_info
if "%choice%"=="5" goto configure
if "%choice%"=="6" goto install_deps
if "%choice%"=="7" goto download_model
if "%choice%"=="0" exit
goto menu

:start_worker
cls
echo.
echo ============================================
echo   Starting Z-Image Worker
echo ============================================
echo.

REM Check API_KEY
if "%API_KEY%"=="YOUR_API_KEY_HERE" (
    echo [ERROR] Please configure API Key first!
    echo.
    echo Edit this bat file, modify line 13: API_KEY
    echo Or run option [5] to configure
    echo.
    pause
    goto menu
)

REM Check Python
if not exist "%PYTHON_EXE%" (
    if "%PYTHON_EXE%"=="python.exe" (
        where python >nul 2>&1
        if %errorlevel% neq 0 (
            echo [ERROR] Python not found!
            echo.
            echo Please run option [6] Install Dependencies first
            echo.
            pause
            goto menu
        )
    ) else (
        echo [ERROR] Python not found: %PYTHON_EXE%
        echo.
        echo Please run option [6] Install Dependencies first
        echo.
        pause
        goto menu
    )
)

REM Check worker.py
if not exist "%WORKER_DIR%\worker.py" (
    echo [ERROR] worker.py not found
    echo.
    pause
    goto menu
)

REM Check .env config
if not exist "%WORKER_DIR%\.env" (
    echo [WARNING] Config file not found, please configure first
    echo.
    goto configure
)
set "EMBEDDED_API_ENABLED=true"
for /f "tokens=2 delims==" %%a in ('findstr "WORKER_ENABLE_OPENAI_COMPAT_API" "%WORKER_DIR%\.env" 2^>nul') do set "EMBEDDED_API_ENABLED=%%a"

REM Prevent launching a second standalone model holder
wmic process where "commandline like '%%openai_compat_server.py%%'" get processid,commandline 2>nul | findstr /i "openai_compat_server" >nul
if %errorlevel%==0 (
    echo [WARNING] Standalone OpenAI Image API is already running.
    echo.
    echo Worker now embeds the OpenAI-compatible API and shares the same model instance.
    echo Stop the standalone API first, otherwise both processes will load their own copy.
    echo.
    pause
    goto menu
)

echo Starting Worker...
echo.
echo   Python: %PYTHON_EXE%
echo   Directory: %WORKER_DIR%
echo.

cd /d "%WORKER_DIR%"
start "Z-Image Worker" cmd /k "set HF_HUB_DISABLE_SYMLINKS_WARNING=1 && "%PYTHON_EXE%" worker.py"

echo.
echo [SUCCESS] Worker started in new window!
echo.
if /i not "%EMBEDDED_API_ENABLED%"=="false" (
    echo   OpenAI API is also exposed by Worker: http://localhost:8787/v1
) else (
    echo   Embedded OpenAI API is disabled in .env
)
echo Note: First start needs to load model (1-2 min), please wait.
echo.
pause
goto menu

:stop_worker
cls
echo.
echo ============================================
echo   Stopping Z-Image Worker / API
echo ============================================
echo.

REM Find and stop worker.py process
set "found=0"
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%worker.py%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo Stopping Worker PID: %%a
    taskkill /PID %%a /F >nul 2>&1
    set "found=1"
)

for /f "tokens=2" %%a in ('wmic process where "commandline like '%%openai_compat_server.py%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo Stopping API PID: %%a
    taskkill /PID %%a /F >nul 2>&1
    set "found=1"
)

if "%found%"=="1" (
    echo.
    echo [SUCCESS] Worker/API stopped!
) else (
    echo [INFO] No running Worker/API process found.
)
echo.
pause
goto menu

:check_status
cls
echo.
echo ============================================
echo   Worker Status
echo ============================================
echo.

echo === Worker Process ===
wmic process where "commandline like '%%worker.py%%'" get processid,commandline 2>nul | findstr /i "worker" >nul
if %errorlevel%==0 (
    echo [RUNNING] Worker is running
    echo.
    wmic process where "commandline like '%%worker.py%%'" get processid 2>nul | findstr /r "[0-9]"
) else (
    echo [STOPPED] Worker is not running
)

echo.
echo === OpenAI Image API ===
set "EMBEDDED_API_ENABLED=true"
if exist "%WORKER_DIR%\.env" (
    for /f "tokens=2 delims==" %%a in ('findstr "WORKER_ENABLE_OPENAI_COMPAT_API" "%WORKER_DIR%\.env" 2^>nul') do set "EMBEDDED_API_ENABLED=%%a"
)
wmic process where "commandline like '%%openai_compat_server.py%%'" get processid,commandline 2>nul | findstr /i "openai_compat_server" >nul
if %errorlevel%==0 (
    echo [RUNNING] OpenAI Image API is running
    echo.
    wmic process where "commandline like '%%openai_compat_server.py%%'" get processid 2>nul | findstr /r "[0-9]"
) else (
    wmic process where "commandline like '%%worker.py%%'" get processid,commandline 2>nul | findstr /i "worker.py" >nul
    if %errorlevel%==0 (
        if /i not "%EMBEDDED_API_ENABLED%"=="false" (
            echo [RUNNING] OpenAI Image API is embedded in Worker ^(shared model^)
            echo   URL: http://localhost:8787/v1
        ) else (
            echo [DISABLED] OpenAI Image API is disabled in Worker config
        )
    ) else (
        echo [STOPPED] OpenAI Image API is not running
    )
)

echo.
echo === Configuration ===
if exist "%WORKER_DIR%\.env" (
    echo Config file: %WORKER_DIR%\.env
    echo.
    type "%WORKER_DIR%\.env" | findstr /v "API_KEY"
    echo WORKER_API_KEY=****** (hidden)
) else (
    echo [WARNING] Config file not found
)

echo.
pause
goto menu

:gpu_info
cls
echo.
echo ============================================
echo   GPU Information
echo ============================================
echo.

where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    nvidia-smi
) else (
    echo [ERROR] nvidia-smi not found
    echo.
    echo Please install NVIDIA driver and CUDA.
    echo Download: https://developer.nvidia.com/cuda-downloads
)

echo.
pause
goto menu

:configure
cls
echo.
echo ============================================
echo   Configure Worker
echo ============================================
echo.

REM Check API_KEY
if "%API_KEY%"=="YOUR_API_KEY_HERE" (
    echo [WARNING] Worker API Key not configured.
    echo.
    echo Remote polling Worker mode will not work until you fill in line 13.
    echo OpenAI Image API mode can still be configured and used locally.
    echo.
)

REM Read existing config or use defaults
set "NEW_ID=worker-%COMPUTERNAME%"
set "NEW_NAME=%COMPUTERNAME%"

if exist "%WORKER_DIR%\.env" (
    for /f "tokens=2 delims==" %%a in ('findstr "WORKER_ID" "%WORKER_DIR%\.env" 2^>nul') do set "NEW_ID=%%a"
    for /f "tokens=2 delims==" %%a in ('findstr "WORKER_NAME" "%WORKER_DIR%\.env" 2^>nul') do set "NEW_NAME=%%a"
)

echo Enter Worker config (press Enter for default):
echo.

set /p "NEW_ID=Worker ID [%NEW_ID%]: " || set "NEW_ID=%NEW_ID%"
set /p "NEW_NAME=Worker Name [%NEW_NAME%]: " || set "NEW_NAME=%NEW_NAME%"

echo.
echo Saving config...

REM Create directories
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

REM Write config file
(
echo # Z-Image Worker Configuration
echo # Generated: %DATE% %TIME%
echo.
echo # Worker Identity
echo WORKER_ID=%NEW_ID%
echo WORKER_NAME=%NEW_NAME%
echo.
echo # Server Connection
echo REMOTE_API_BASE=%API_BASE%
echo WORKER_API_KEY=%API_KEY%
echo.
echo # Model Configuration
echo MODEL_ID=Tongyi-MAI/Z-Image-Turbo
echo DEVICE=cuda
echo MULTI_GPU_MODE=auto
echo MULTI_GPU_DEVICES=
echo GPU_MEMORY_RESERVE_GB=0.5
echo USE_CPU_OFFLOAD=true
echo WORKER_ENABLE_OPENAI_COMPAT_API=true
echo.
echo # OpenAI-Compatible Image API
echo OPENAI_COMPAT_HOST=0.0.0.0
echo OPENAI_COMPAT_PORT=8787
echo OPENAI_COMPAT_API_KEY=
echo OPENAI_COMPAT_MODEL_NAME=Tongyi-MAI/Z-Image-Turbo
echo OPENAI_COMPAT_PUBLIC_BASE_URL=
echo OPENAI_COMPAT_DEFAULT_RESPONSE_FORMAT=url
echo OPENAI_COMPAT_MAX_IMAGES_PER_REQUEST=1
echo.
echo # Timing Configuration
echo HEARTBEAT_INTERVAL=10
echo POLL_INTERVAL=2
echo JOB_TIMEOUT=300
echo.
echo # Local Backup
echo LOCAL_BACKUP_ROOT=%BACKUP_DIR%
) > "%WORKER_DIR%\.env"

echo.
echo [SUCCESS] Config saved!
echo.
echo   Worker ID: %NEW_ID%
echo   Worker Name: %NEW_NAME%
echo   Server: %API_BASE%
echo.
pause
goto menu

:install_deps
cls
echo.
echo ============================================
echo   Install/Update Dependencies
echo ============================================
echo.

REM Check Python 3.11 (required for PyTorch)
set "PY311="
for /f "delims=" %%i in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PY311=%%i"

if "%PY311%"=="" (
    echo [ERROR] Python 3.11 not found!
    echo.
    echo Please install Python 3.11:
    echo   winget install Python.Python.3.11 --source winget
    echo.
    echo Note: Python 3.13 does NOT support PyTorch yet
    echo.
    pause
    goto menu
)

echo Detected Python 3.11:
py -3.11 --version
echo Path: %PY311%
echo.

REM Set venv directory
set "INSTALL_VENV=%WORKER_DIR%\venv"

REM Check venv
if not exist "%INSTALL_VENV%\Scripts\python.exe" (
    echo.
    echo Creating virtual environment...
    echo Directory: %INSTALL_VENV%
    echo.
    py -3.11 -m venv "%INSTALL_VENV%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        echo.
        echo If using Python 3.13, please install Python 3.11:
        echo   winget install Python.Python.3.11 --source winget
        echo.
        pause
        goto menu
    )
    echo [SUCCESS] Virtual environment created
    echo.
    
    set "PYTHON_EXE=%INSTALL_VENV%\Scripts\python.exe"
)

echo Installing dependencies...
echo This may take 5-10 minutes, please wait...
echo.

REM Activate venv and install
call "%INSTALL_VENV%\Scripts\activate.bat"

echo --------------------------------------------
echo [1/3] Installing PyTorch (CUDA 12.1)
echo --------------------------------------------
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] PyTorch install failed!
    echo Possible cause: Python version incompatible
    echo.
    pause
    goto menu
)

echo.
echo --------------------------------------------
echo [2/3] Installing diffusers (from GitHub)
echo --------------------------------------------
echo.
echo This step may take a few minutes, please wait...
pip install --progress-bar on git+https://github.com/huggingface/diffusers
echo [2/3] Done!

echo.
echo --------------------------------------------
echo [3/3] Installing other dependencies
echo --------------------------------------------
echo.
pip install --progress-bar on transformers accelerate safetensors sentencepiece huggingface_hub Pillow httpx python-dotenv
echo [3/3] Done!

echo.
echo ============================================
echo [DONE] Dependencies installed!
echo ============================================
echo.
echo Virtual env: %INSTALL_VENV%
echo.
echo Next step: Run option [7] to download model
echo.
pause
goto menu

:download_model
cls
echo.
echo ============================================
echo   Download/Update Model
echo ============================================
echo.

if not exist "%WORKER_DIR%\venv\Scripts\python.exe" (
    echo [ERROR] Please run option [6] Install Dependencies first!
    echo.
    pause
    goto menu
)

echo ************************************************************
echo *  IMPORTANT: Enable Windows Developer Mode first!         *
echo *                                                          *
echo *  Settings - Privacy and Security - Developer Options     *
echo *  Turn on "Developer Mode"                                *
echo *                                                          *
echo *  Otherwise model download will FAIL!                     *
echo ************************************************************
echo.
echo Model Info:
echo   Name: Z-Image-Turbo (Tongyi Wanxiang)
echo   Size: About 25GB
echo   Source: HuggingFace
echo.
echo Tip: If download is slow, set mirror:
echo   set HF_ENDPOINT=https://hf-mirror.com
echo.
echo Download will run in a new window.
echo.
set /p confirm=Developer Mode enabled? Start download? [Y/N]: 

if /i "%confirm%"=="Y" (
    echo.
    echo Starting download...
    start "Model Download" cmd /k "set HF_HUB_DISABLE_SYMLINKS_WARNING=1 && "%WORKER_DIR%\venv\Scripts\python.exe" "%WORKER_DIR%\download_model.py""
    echo.
    echo [INFO] Download started in new window!
    echo.
    echo After download, run option [1] to start Worker
)

echo.
pause
goto menu

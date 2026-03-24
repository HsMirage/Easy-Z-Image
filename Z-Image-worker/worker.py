# -*- coding: utf-8 -*-
"""
Z-Image Worker 主服务

功能：
1. 定时发送心跳
2. 拉取待处理任务
3. 调用 Z-Image 生成图片
4. 上传结果到远程服务器
5. 本地备份
"""
import sys
import os
import time
import signal
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

# Windows UTF-8 支持
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import (
    WORKER_ID, WORKER_NAME,
    HEARTBEAT_INTERVAL, POLL_INTERVAL, JOB_TIMEOUT,
    LOCAL_BACKUP_ROOT, GPU_INFO, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    OPENAI_COMPAT_HOST, OPENAI_COMPAT_PORT,
    PRELOAD_MODEL_ON_START, WORKER_ENABLE_OPENAI_COMPAT_API,
    WORKER_UNLOAD_MODEL_AFTER_JOB, WORKER_UNLOAD_MODEL_ON_ERROR,
)
from generator import GenerationParams, get_generation_lock, get_generator
from api_client import get_api_client
from openai_compat_server import CompatHTTPServer, create_server


class Worker:
    """Worker 主服务"""
    
    def __init__(self):
        self.running = False
        self.current_job_id: Optional[str] = None
        self.status = "idle"  # idle / busy
        
        self.generator = get_generator()
        self.runtime_lock = get_generation_lock()
        self.api_client = get_api_client()
        
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.compat_api_server: Optional[CompatHTTPServer] = None
        self.compat_api_thread: Optional[threading.Thread] = None
        
    def start(self):
        """启动 Worker"""
        print("=" * 60)
        print(f"  Z-Image Worker")
        print(f"  ID: {WORKER_ID}")
        print(f"  Name: {WORKER_NAME}")
        print("=" * 60)

        self.running = True
        
        # 启动心跳线程
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        self._start_embedded_openai_api()
        
        # 主循环：拉取任务
        if PRELOAD_MODEL_ON_START:
            print("\n[Worker] Preloading model onto GPU(s) before polling...")
            with self.runtime_lock:
                self.status = "busy"
                self.generator.load_model()
                self.status = "idle"
            print(
                f"[Worker] Preload complete: mode={self.generator.placement.mode}, "
                f"dtype={self.generator.placement.dtype_name}, "
                f"execution={self.generator.placement.execution_device}, "
                f"gpus={self.generator.placement.gpu_ids}"
            )
        else:
            print("\n[Worker] Model loading is deferred until the first job")
        print(f"[Worker] Keep model loaded after job: {not WORKER_UNLOAD_MODEL_AFTER_JOB}")
        print(f"[Worker] Started! Polling interval: {POLL_INTERVAL}s")
        self._main_loop()
        
    def stop(self):
        """停止 Worker"""
        if self.status == "offline":
            return

        print("\n[Worker] Stopping...")
        self.running = False
        self._stop_embedded_openai_api()
        
        # 发送离线心跳
        self.status = "offline"
        self.api_client.send_heartbeat("offline", None, GPU_INFO)
        
        self.api_client.close()
        self.generator.unload_model()
        
        print("[Worker] Stopped")
        
    def _heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            try:
                gpu_status = self.generator.get_gpu_status()
                effective_status = self.status
                if effective_status != "offline" and self.runtime_lock.locked():
                    effective_status = "busy"
                result = self.api_client.send_heartbeat(
                    effective_status,
                    self.current_job_id,
                    {**GPU_INFO, **gpu_status}
                )
                if result.get("success") is False:
                    print(f"[Heartbeat] Warning: {result.get('error')}")
            except Exception as e:
                print(f"[Heartbeat] Error: {e}")
                
            time.sleep(HEARTBEAT_INTERVAL)
            
    def _main_loop(self):
        """主循环：拉取和处理任务"""
        while self.running:
            try:
                if self.status == "idle" and not self.runtime_lock.locked():
                    job = self.api_client.fetch_next_job()
                    if job:
                        self._process_job(job)
                        
            except Exception as e:
                print(f"[Worker] Error in main loop: {e}")
                
            time.sleep(POLL_INTERVAL)
            
    def _process_job(self, job: dict):
        """处理单个任务"""
        job_id = job.get("id")
        user_id = job.get("user_id")
        
        print(f"\n[Job] Processing job: {job_id}")
        print(f"[Job] User: {user_id}")
        
        self.current_job_id = job_id
        self.status = "busy"
        
        with self.runtime_lock:
            try:
                # 更新状态为 running
                self.api_client.update_job_status(job_id, "running")
                
                # 构建生成参数
                params = GenerationParams(
                    prompt=job.get("prompt", ""),
                    width=job.get("width", DEFAULT_WIDTH),
                    height=job.get("height", DEFAULT_HEIGHT),
                    steps=job.get("steps", 9),
                    seed=job.get("seed", -1),
                )
                
                # 生成图片
                image, metadata = self.generator.generate(params)
                
                # 保存到本地备份
                backup_path = self._save_local_backup(job_id, user_id, image)
                print(f"[Job] Local backup: {backup_path}")
                
                # 上传结果
                success = self.api_client.upload_result(job_id, backup_path, metadata)
                
                if success:
                    self.api_client.update_job_status(job_id, "done")
                    print(f"[Job] Completed: {job_id}")
                else:
                    self.api_client.update_job_status(job_id, "failed", "Failed to upload result")
                    print(f"[Job] Upload failed: {job_id}")
                    
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[Job] Error: {error_msg}")
                traceback.print_exc()
                self.api_client.update_job_status(job_id, "failed", error_msg)
                if WORKER_UNLOAD_MODEL_ON_ERROR:
                    print("[Job] Unloading model after error to release VRAM")
                    self.generator.unload_model()
                
            finally:
                if WORKER_UNLOAD_MODEL_AFTER_JOB and self.generator.loaded:
                    print("[Job] Unloading model after job")
                    self.generator.unload_model()
                else:
                    if self.generator.loaded:
                        print("[Job] Keeping model resident on GPU(s)")
                    self.generator.release_memory()
                self.current_job_id = None
                self.status = "idle"

    def _start_embedded_openai_api(self) -> None:
        """在 Worker 进程内启动 OpenAI 兼容 API，共用同一份生成器。"""
        if not WORKER_ENABLE_OPENAI_COMPAT_API:
            print("[Worker] Embedded OpenAI-compatible API is disabled by config")
            return

        try:
            _, self.compat_api_server = create_server()
        except OSError as exc:
            self.compat_api_server = None
            print(
                "[Worker] Embedded OpenAI-compatible API failed to bind "
                f"http://{OPENAI_COMPAT_HOST}:{OPENAI_COMPAT_PORT}: {exc}"
            )
            print("[Worker] If a standalone `openai_compat_server.py` is already running, stop it first.")
            return

        self.compat_api_thread = threading.Thread(
            target=self.compat_api_server.serve_forever,
            daemon=True,
            name="zimage-openai-compat",
        )
        self.compat_api_thread.start()
        print(
            "[Worker] Embedded OpenAI-compatible API is sharing the in-process generator at "
            f"http://{OPENAI_COMPAT_HOST}:{OPENAI_COMPAT_PORT}/v1"
        )

    def _stop_embedded_openai_api(self) -> None:
        """停止 Worker 内嵌的 OpenAI 兼容 API。"""
        if self.compat_api_server is None:
            return

        print("[Worker] Stopping embedded OpenAI-compatible API...")
        self.compat_api_server.shutdown()
        self.compat_api_server.server_close()
        if self.compat_api_thread is not None and self.compat_api_thread.is_alive():
            self.compat_api_thread.join(timeout=5)
        self.compat_api_server = None
        self.compat_api_thread = None
            
    def _save_local_backup(self, job_id: str, user_id: str, image) -> Path:
        """
        保存本地备份
        
        目录结构: storage_root/user_id/YYYY-MM-DD/job_id.png
        """
        today = datetime.now().strftime("%Y-%m-%d")
        backup_dir = LOCAL_BACKUP_ROOT / str(user_id) / today
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_path = backup_dir / f"{job_id}.png"
        image.save(backup_path)
        
        return backup_path


def main():
    """主入口"""
    worker = Worker()
    
    # 信号处理
    def signal_handler(signum, frame):
        print(f"\n[Signal] Received signal {signum}")
        worker.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        worker.start()
    except KeyboardInterrupt:
        worker.stop()
    except Exception as e:
        print(f"[Fatal] {e}")
        worker.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()




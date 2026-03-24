# -*- coding: utf-8 -*-
"""远程 API 客户端"""
import httpx
from typing import Optional
from pathlib import Path

from config import REMOTE_API_BASE, API_KEY, WORKER_ID


class APIClient:
    """与远程后端通信的客户端"""
    
    def __init__(self):
        self.base_url = REMOTE_API_BASE.rstrip("/")
        # 认证 headers（不设置 Content-Type，让 httpx 自动处理）
        self.auth_headers = {
            "X-Worker-ID": WORKER_ID,
            "X-API-Key": API_KEY,
        }
        self.client = httpx.Client(timeout=60.0, headers=self.auth_headers)
        
    def send_heartbeat(self, status: str, current_job_id: Optional[str], gpu_info: dict) -> dict:
        """
        发送心跳
        
        Args:
            status: idle / busy
            current_job_id: 当前正在处理的任务 ID
            gpu_info: GPU 信息
        """
        payload = {
            "worker_id": WORKER_ID,
            "status": status,
            "current_job_id": current_job_id,
            "gpu_info": gpu_info,
        }
        
        try:
            resp = self.client.post(f"{self.base_url}/api/workers/heartbeat", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[API] Heartbeat failed: {e}")
            return {"success": False, "error": str(e)}
    
    def fetch_next_job(self) -> Optional[dict]:
        """
        拉取下一个待处理任务
        
        Returns:
            任务字典或 None
        """
        try:
            resp = self.client.get(f"{self.base_url}/api/workers/{WORKER_ID}/next-job")
            if resp.status_code == 204:
                # 无任务
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[API] Fetch job failed: {e}")
            return None
    
    def update_job_status(self, job_id: str, status: str, error_message: Optional[str] = None) -> bool:
        """更新任务状态"""
        payload = {
            "status": status,
            "error_message": error_message,
        }
        
        try:
            resp = self.client.patch(f"{self.base_url}/api/jobs/{job_id}/status", json=payload)
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"[API] Update job status failed: {e}")
            return False
    
    def upload_result(self, job_id: str, image_path: Path, metadata: dict) -> bool:
        """
        上传生成结果
        
        Args:
            job_id: 任务 ID
            image_path: 本地图片路径
            metadata: 生成元数据
        """
        import json
        
        try:
            with open(image_path, "rb") as f:
                files = {"image": (image_path.name, f, "image/png")}
                data = {"metadata": json.dumps(metadata)}
                
                # 使用独立的客户端上传，避免 headers 冲突
                with httpx.Client(timeout=120.0) as upload_client:
                    resp = upload_client.post(
                        f"{self.base_url}/api/jobs/{job_id}/result",
                        files=files,
                        data=data,
                        headers={
                            "X-Worker-ID": WORKER_ID,
                            "X-API-Key": API_KEY,
                        },
                    )
                    print(f"[API] Upload response: {resp.status_code}")
                    if resp.status_code != 200:
                        print(f"[API] Response: {resp.text[:500]}")
                    resp.raise_for_status()
                    return True
        except Exception as e:
            print(f"[API] Upload result failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close(self):
        """关闭客户端"""
        self.client.close()


# 全局单例
_client: Optional[APIClient] = None


def get_api_client() -> APIClient:
    """获取 API 客户端单例"""
    global _client
    if _client is None:
        _client = APIClient()
    return _client



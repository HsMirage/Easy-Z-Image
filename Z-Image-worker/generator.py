# -*- coding: utf-8 -*-
"""Z-Image 图像生成器"""
import sys
import os
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Windows UTF-8 支持
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
from PIL import Image

from config import (
    MODEL_ID, DEVICE, USE_CPU_OFFLOAD,
    DEFAULT_STEPS, MAX_WIDTH, MAX_HEIGHT
)


@dataclass
class GenerationParams:
    """生成参数"""
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = DEFAULT_STEPS
    seed: int = -1  # -1 表示随机
    
    def validate(self) -> None:
        """验证参数"""
        if not self.prompt or not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if self.width > MAX_WIDTH or self.height > MAX_HEIGHT:
            raise ValueError(f"Resolution exceeds limit: max {MAX_WIDTH}x{MAX_HEIGHT}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid resolution")


class ZImageGenerator:
    """Z-Image 图像生成器"""
    
    def __init__(self):
        self.pipe = None
        self.device = DEVICE
        self.loaded = False
        
    def load_model(self) -> None:
        """加载模型"""
        if self.loaded:
            return
            
        from config import MODEL_REVISION
        revision = MODEL_REVISION.strip() if MODEL_REVISION else None
        rev_info = f" (revision: {revision})" if revision else " (latest)"
        print(f"[Generator] Loading model: {MODEL_ID}{rev_info}")
        print(f"[Generator] Device: {self.device}, CPU Offload: {USE_CPU_OFFLOAD}")
        
        from diffusers import ZImagePipeline
        
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if revision:
            load_kwargs["revision"] = revision
            load_kwargs["local_files_only"] = True  # 指定版本时只用本地缓存
        
        self.pipe = ZImagePipeline.from_pretrained(MODEL_ID, **load_kwargs)
        
        # 8-10GB 显存使用 sequential CPU offload 节省显存
        print("[Generator] Enabling sequential CPU offload...")
        self.pipe.enable_sequential_cpu_offload()
            
        self.loaded = True
        print("[Generator] Model loaded successfully!")
        
    def unload_model(self) -> None:
        """卸载模型释放显存"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Generator] Model unloaded")
            
    def generate(self, params: GenerationParams) -> tuple[Image.Image, dict]:
        """
        生成图像
        
        Returns:
            tuple: (生成的图像, 元数据dict)
        """
        params.validate()
        
        if not self.loaded:
            self.load_model()
            
        # 设置随机种子
        actual_seed = params.seed if params.seed >= 0 else int(time.time() * 1000) % (2**32)
        generator = torch.Generator(self.device).manual_seed(actual_seed)
        
        print(f"[Generator] Generating: {params.width}x{params.height}, steps={params.steps}, seed={actual_seed}")
        print(f"[Generator] Prompt: {params.prompt[:100]}...")
        
        start_time = time.time()
        
        # 生成图像
        # 注意：Z-Image-Turbo 是蒸馏模型，guidance_scale 必须为 0
        result = self.pipe(
            prompt=params.prompt,
            height=params.height,
            width=params.width,
            num_inference_steps=params.steps,
            guidance_scale=0.0,
            generator=generator,
        )
        
        image = result.images[0]
        elapsed = time.time() - start_time
        
        print(f"[Generator] Generated in {elapsed:.2f}s")
        
        metadata = {
            "seed": actual_seed,
            "steps": params.steps,
            "width": params.width,
            "height": params.height,
            "elapsed_seconds": round(elapsed, 2),
            "model": MODEL_ID,
        }
        
        return image, metadata
    
    def get_gpu_status(self) -> dict:
        """获取 GPU 状态（不包含 name，让 config.py 的 GPU_INFO.name 生效）"""
        if not torch.cuda.is_available():
            return {"available": False}
            
        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),  # 原始设备名，不覆盖自定义 name
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
            "memory_used_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "memory_cached_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
        }


# 全局单例
_generator: Optional[ZImageGenerator] = None


def get_generator() -> ZImageGenerator:
    """获取生成器单例"""
    global _generator
    if _generator is None:
        _generator = ZImageGenerator()
    return _generator


if __name__ == "__main__":
    # 测试
    gen = get_generator()
    gen.load_model()
    
    params = GenerationParams(
        prompt="a cute orange cat sleeping in warm sunlight, photorealistic",
        width=1024,
        height=1024,
        seed=42,
    )
    
    image, meta = gen.generate(params)
    
    output_path = Path("../outputs/test_generator.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    print(f"Saved to: {output_path}")
    print(f"Metadata: {meta}")


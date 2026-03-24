# -*- coding: utf-8 -*-
"""Z-Image 图像生成器"""
import sys
import os
import time
import gc
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, replace

# Load .env before importing torch so allocator settings take effect.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Windows UTF-8 支持
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
from PIL import Image

from config import (
    MODEL_ID, MODEL_REVISION, DEVICE, TORCH_DTYPE, MULTI_GPU_MODE, MULTI_GPU_DEVICES,
    GPU_MEMORY_RESERVE_GB, MULTI_GPU_VAE_DECODE_RESERVE_GB, MULTI_GPU_ACTIVATION_RESERVE_GB, USE_CPU_OFFLOAD,
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_STEPS, MAX_WIDTH, MAX_HEIGHT
)


@dataclass
class GenerationParams:
    """生成参数"""
    prompt: str
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
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


@dataclass
class ModelPlacement:
    """当前模型放置策略"""
    mode: str = "single"
    execution_device: str = DEVICE
    dtype_name: str = "fp32"
    text_encoder_device: Optional[str] = None
    vae_device: Optional[str] = None
    transformer_device_map: Optional[dict] = None
    transformer_max_memory: Optional[dict] = None
    gpu_ids: list[int] = field(default_factory=list)


class ZImageGenerator:
    """Z-Image 图像生成器"""
    
    def __init__(self):
        self.pipe = None
        self.device = DEVICE
        self.loaded = False
        self.placement = ModelPlacement(execution_device=DEVICE, dtype_name=self._dtype_name(torch.float32))
        self._size_estimates_gib: dict[tuple[str, str], float] = {}
        self._dtype_override: Optional[str] = None
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def _dtype_name(dtype: torch.dtype) -> str:
        """Readable precision label for logs/metadata."""
        if dtype == torch.bfloat16:
            return "bf16"
        if dtype == torch.float16:
            return "fp16"
        if dtype == torch.float32:
            return "fp32"
        return str(dtype).replace("torch.", "")

    def _torch_dtype(self):
        """Pick a stable dtype for the active hardware."""
        choice = (self._dtype_override or TORCH_DTYPE or "auto").strip().lower()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            if choice == "auto":
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            if choice in {"bf16", "bfloat16"}:
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("TORCH_DTYPE=bf16 requested, but this GPU does not support bfloat16")
                return torch.bfloat16
            if choice in {"fp16", "float16", "half"}:
                return torch.float16
            if choice in {"fp32", "float32"}:
                return torch.float32
            raise ValueError(f"Unsupported TORCH_DTYPE: {choice}")

        if choice not in {"auto", "fp32", "float32"}:
            print(f"[Generator] TORCH_DTYPE={choice} ignored on non-CUDA device, using fp32")
        return torch.float32

    def _model_source_kwargs(self) -> dict:
        """统一的模型来源参数。"""
        kwargs = {}
        if MODEL_REVISION:
            kwargs["revision"] = MODEL_REVISION
            kwargs["local_files_only"] = True
        return kwargs

    def _multi_gpu_enabled(self) -> bool:
        """是否尝试单进程多卡分片加载。"""
        if self.device != "cuda":
            return False
        if MULTI_GPU_MODE in {"off", "disabled", "false", "0"}:
            return False
        if not torch.cuda.is_available():
            return False
        return len(self._get_multi_gpu_ids()) >= 2

    def _multi_gpu_required(self) -> bool:
        """多卡失败时是否直接报错。"""
        return MULTI_GPU_MODE in {"force", "required"}

    def _get_multi_gpu_ids(self) -> list[int]:
        """解析要参与分片的 GPU。"""
        if not torch.cuda.is_available():
            return []

        device_count = torch.cuda.device_count()
        if device_count < 2:
            return []

        if not MULTI_GPU_DEVICES:
            return list(range(device_count))

        gpu_ids = []
        for part in MULTI_GPU_DEVICES.split(","):
            part = part.strip()
            if not part:
                continue
            gpu_id = int(part)
            if gpu_id < 0 or gpu_id >= device_count:
                raise ValueError(
                    f"Invalid MULTI_GPU_DEVICES entry: {gpu_id}, available range is 0-{device_count - 1}"
                )
            if gpu_id not in gpu_ids:
                gpu_ids.append(gpu_id)

        return gpu_ids

    def _gpu_total_gib(self, gpu_id: int) -> float:
        """获取 GPU 总显存（GiB）。"""
        return torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3

    def _gpu_free_gib(self, gpu_id: int) -> float:
        """获取 GPU 当前空闲显存（GiB）。"""
        free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
        return free_bytes / 1024**3

    def _gpu_budget_gib(self, gpu_id: int) -> float:
        """优先吃满本机 GPU；只有显存明显被外部进程占用时才收缩预算。"""
        total_gib = self._gpu_total_gib(gpu_id)
        free_gib = self._gpu_free_gib(gpu_id)
        unavailable_gib = max(total_gib - free_gib, 0.0)
        if unavailable_gib <= max(0.75, GPU_MEMORY_RESERVE_GB):
            return total_gib
        return free_gib

    def _model_load_kwargs(
        self,
        dtype: torch.dtype,
        *,
        device_map=None,
        max_memory=None,
    ) -> dict:
        """统一模型加载参数，尽量减少 CPU/RAM 压力。"""
        kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        if max_memory is not None:
            # offload_state_dict only for transformer sharding (max_memory set);
            # single-device device_map={"":gpu} loads don't need it and it wastes VRAM
            kwargs["offload_state_dict"] = True
            kwargs["max_memory"] = max_memory
        kwargs.update(self._model_source_kwargs())
        return kwargs

    def _estimate_component_size_gib(self, component_name: str, dtype: torch.dtype) -> float:
        """估算组件按目标 dtype 加载后的显存占用。"""
        cache_key = (component_name, str(dtype))
        if cache_key in self._size_estimates_gib:
            return self._size_estimates_gib[cache_key]

        from accelerate import init_empty_weights

        source_kwargs = self._model_source_kwargs()

        if component_name == "text_encoder":
            from transformers import AutoConfig, AutoModel

            config = AutoConfig.from_pretrained(MODEL_ID, subfolder="text_encoder", **source_kwargs)
            with init_empty_weights():
                model = AutoModel.from_config(config)
        elif component_name == "transformer":
            from diffusers import ZImageTransformer2DModel

            config = ZImageTransformer2DModel.load_config(MODEL_ID, subfolder="transformer", **source_kwargs)
            with init_empty_weights():
                model = ZImageTransformer2DModel.from_config(config)
        elif component_name == "vae":
            from diffusers import AutoencoderKL

            config = AutoencoderKL.load_config(MODEL_ID, subfolder="vae", **source_kwargs)
            with init_empty_weights():
                model = AutoencoderKL.from_config(config)
        else:
            raise ValueError(f"Unknown component: {component_name}")

        element_count = sum(param.numel() for param in model.parameters())
        element_count += sum(buffer.numel() for buffer in model.buffers())
        size_gib = element_count * torch.tensor([], dtype=dtype).element_size() / 1024**3

        # Qwen3-based text encoder: actual VRAM after CPU->GPU load matches
        # the theoretical parameter size (no tied-weight savings observed in
        # practice).  No scaling factor needed.
        _ = component_name  # no per-component adjustment

        self._size_estimates_gib[cache_key] = size_gib
        del model
        return size_gib

    def _build_multi_gpu_plan(self, dtype: torch.dtype) -> ModelPlacement:
        """构建单进程多卡分片计划。"""
        gpu_ids = self._get_multi_gpu_ids()
        gpu_memory = {
            gpu_id: {
                "free_gib": self._gpu_free_gib(gpu_id),
                "total_gib": self._gpu_total_gib(gpu_id),
                "budget_gib": self._gpu_budget_gib(gpu_id),
            }
            for gpu_id in gpu_ids
        }

        text_encoder_gib = self._estimate_component_size_gib("text_encoder", dtype)
        vae_gib = self._estimate_component_size_gib("vae", dtype)
        transformer_gib = self._estimate_component_size_gib("transformer", dtype)
        vae_decode_reserve_gib = max(0.0, MULTI_GPU_VAE_DECODE_RESERVE_GB)

        best_text_encoder_gpu = None
        best_vae_gpu = None
        best_transformer_max_memory = None
        best_score = None
        best_transformer_available_gib = 0.0

        # 穷举文本编码器和 VAE 落卡位置，同时在 VAE 所在卡额外预留 decode headroom，
        # 以便 22 GiB 级别的双卡也能尽量避免把 text encoder/transformer 挪回 CPU。
        for text_encoder_gpu in gpu_ids:
            for vae_gpu in gpu_ids:
                candidate_max_memory = {}
                candidate_available_gib = 0.0

                for gpu_id in gpu_ids:
                    pinned_gib = 0.0
                    if gpu_id == text_encoder_gpu:
                        pinned_gib += text_encoder_gib
                    if gpu_id == vae_gpu:
                        pinned_gib += vae_gib + vae_decode_reserve_gib

                    available_gib = gpu_memory[gpu_id]["budget_gib"] - pinned_gib - GPU_MEMORY_RESERVE_GB
                    if available_gib <= 0:
                        candidate_max_memory = None
                        break

                    candidate_available_gib += available_gib
                    candidate_max_memory[gpu_id] = available_gib

                if not candidate_max_memory or len(candidate_max_memory) < 2:
                    continue

                effective_max_gib = sum(
                    max(v - MULTI_GPU_ACTIVATION_RESERVE_GB, 0.5)
                    for v in candidate_max_memory.values()
                )
                if effective_max_gib <= transformer_gib:
                    continue

                score = (
                    candidate_available_gib,
                    # Maximise the smaller GPU's budget: this ensures both GPUs
                    # have enough room for transformer shards and avoids starving
                    # one card (which causes disk offload).
                    min(candidate_max_memory.values()),
                    # Prefer pinning fixed components to the smaller GPU so the
                    # larger GPU's headroom is maximised for transformer shards.
                    sum(
                        -self._gpu_total_gib(g)
                        for g in {text_encoder_gpu, vae_gpu}
                    ),
                    int(text_encoder_gpu != vae_gpu),
                )
                if best_score is None or score > best_score:
                    best_text_encoder_gpu = text_encoder_gpu
                    best_vae_gpu = vae_gpu
                    best_transformer_max_memory = candidate_max_memory
                    best_score = score
                    best_transformer_available_gib = candidate_available_gib

        if best_text_encoder_gpu is None or best_vae_gpu is None or best_transformer_max_memory is None:
            raise RuntimeError(
                "Unable to find a GPU placement that leaves enough VRAM for transformer sharding."
            )

        text_encoder_gpu = best_text_encoder_gpu
        vae_gpu = best_vae_gpu
        transformer_max_memory = {
            gpu_id: int(max(available_gib - MULTI_GPU_ACTIVATION_RESERVE_GB, 0.5) * 1024**3)
            for gpu_id, available_gib in best_transformer_max_memory.items()
        }
        transformer_max_memory_readable = {
            gpu_id: f"{available_gib:.2f}GiB" for gpu_id, available_gib in best_transformer_max_memory.items()
        }

        print("[Generator] Multi-GPU plan:")
        print(f"  GPUs: {gpu_ids}")
        print(f"  Reserve per GPU: {GPU_MEMORY_RESERVE_GB:.2f} GiB")
        print(f"  VAE decode reserve: {vae_decode_reserve_gib:.2f} GiB")
        for gpu_id in gpu_ids:
            print(
                f"  GPU {gpu_id}: free={gpu_memory[gpu_id]['free_gib']:.2f} GiB, "
                f"total={gpu_memory[gpu_id]['total_gib']:.2f} GiB, "
                f"budget={gpu_memory[gpu_id]['budget_gib']:.2f} GiB"
            )
        print(f"  Text encoder: cuda:{text_encoder_gpu} (~{text_encoder_gib:.2f} GiB)")
        print(f"  VAE: cuda:{vae_gpu} (~{vae_gib:.2f} GiB)")
        print(f"  Transformer target size: ~{transformer_gib:.2f} GiB")
        print(f"  Transformer available: ~{best_transformer_available_gib:.2f} GiB")
        print(f"  Transformer max_memory: {transformer_max_memory_readable}")

        return ModelPlacement(
            mode="multi_gpu",
            execution_device=f"cuda:{text_encoder_gpu}",
            dtype_name=self._dtype_name(dtype),
            text_encoder_device=f"cuda:{text_encoder_gpu}",
            vae_device=f"cuda:{vae_gpu}",
            transformer_max_memory=transformer_max_memory,
            gpu_ids=gpu_ids,
        )

    def _transformer_modules_for_device(self, device_str: str) -> list[tuple[str, torch.nn.Module]]:
        """返回当前 transformer 上落在指定设备的模块。"""
        if self.pipe is None or getattr(self.pipe, "transformer", None) is None:
            return []
        if not self.placement.transformer_device_map:
            return []

        named_modules = dict(self.pipe.transformer.named_modules())
        result = []
        seen = set()
        for module_name, mapped_device in self.placement.transformer_device_map.items():
            normalized_device = f"cuda:{mapped_device}" if isinstance(mapped_device, int) else str(mapped_device)
            if normalized_device != device_str:
                continue
            module = named_modules.get(module_name)
            if module is None:
                continue
            module_id = id(module)
            if module_id in seen:
                continue
            seen.add(module_id)
            result.append((module_name, module))
        return result

    def _offload_transformer_for_vae_decode(self) -> list[tuple[str, torch.nn.Module, str]]:
        """在 VAE decode 前临时挪走同卡的 transformer 模块，给解码腾显存。"""
        vae_device = self.placement.vae_device
        if not vae_device or not vae_device.startswith("cuda:"):
            return []

        modules = self._transformer_modules_for_device(vae_device)
        if not modules:
            return []

        moved_modules = []
        for module_name, module in modules:
            module.to("cpu")
            moved_modules.append((module_name, module, vae_device))

        gc.collect()
        for gpu_id in range(torch.cuda.device_count()):
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()

        print(
            f"[Generator] Offloaded {len(moved_modules)} transformer modules from {vae_device} "
            "before VAE decode"
        )
        return moved_modules

    def _discard_transformer_for_vae_decode(self) -> bool:
        """在禁用 CPU offload 时，直接把同卡 transformer 模块丢到 meta 释放显存。"""
        vae_device = self.placement.vae_device
        if not vae_device or not vae_device.startswith("cuda:"):
            return False

        modules = self._transformer_modules_for_device(vae_device)
        if not modules:
            return False

        for _, module in modules:
            module.to_empty(device=torch.device("meta"))

        gc.collect()
        for gpu_id in range(torch.cuda.device_count()):
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()

        print(
            f"[Generator] Discarded {len(modules)} transformer modules from {vae_device} "
            "to meta before VAE decode"
        )
        return True

    def _restore_transformer_after_vae_decode(self, moved_modules: list[tuple[str, torch.nn.Module, str]]) -> None:
        """把为 VAE decode 临时挪走的 transformer 模块放回原卡。"""
        if not moved_modules:
            return

        for module_name, module, device_str in moved_modules:
            module.to(device_str)

        print("[Generator] Restored transformer modules after VAE decode")

    def _offload_text_encoder_for_vae_decode(self) -> Optional[str]:
        """在 VAE decode 前临时挪走 text encoder，释放同卡显存。"""
        if self.pipe is None or getattr(self.pipe, "text_encoder", None) is None:
            return None

        text_encoder_device = self.placement.text_encoder_device
        if not text_encoder_device or not text_encoder_device.startswith("cuda:"):
            return None
        if text_encoder_device != self.placement.vae_device:
            return None

        self.pipe.text_encoder.to("cpu")
        gc.collect()
        for gpu_id in range(torch.cuda.device_count()):
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()

        print(f"[Generator] Offloaded text encoder from {text_encoder_device} before VAE decode")
        return text_encoder_device

    def _discard_text_encoder_for_vae_decode(self) -> bool:
        """在禁用 CPU offload 时，直接把同卡 text encoder 丢到 meta 释放显存。"""
        if self.pipe is None or getattr(self.pipe, "text_encoder", None) is None:
            return False

        text_encoder_device = self.placement.text_encoder_device
        if not text_encoder_device or not text_encoder_device.startswith("cuda:"):
            return False
        if text_encoder_device != self.placement.vae_device:
            return False

        self.pipe.text_encoder.to_empty(device=torch.device("meta"))
        gc.collect()
        for gpu_id in range(torch.cuda.device_count()):
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()

        print(f"[Generator] Discarded text encoder from {text_encoder_device} to meta before VAE decode")
        return True

    def _restore_text_encoder_after_vae_decode(self, device_str: Optional[str]) -> None:
        """把为 VAE decode 临时挪走的 text encoder 放回原卡。"""
        if not device_str or self.pipe is None or getattr(self.pipe, "text_encoder", None) is None:
            return

        self.pipe.text_encoder.to(device_str)
        print(f"[Generator] Restored text encoder to {device_str}")

    def _load_model_multi_gpu(self) -> None:
        """按组件手动加载 pipeline，并将 transformer 切到多张 GPU。"""
        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            ZImagePipeline,
            ZImageTransformer2DModel,
        )
        from transformers import AutoModel, AutoTokenizer

        dtype = self._torch_dtype()
        source_kwargs = self._model_source_kwargs()
        plan = self._build_multi_gpu_plan(dtype)

        text_encoder_gpu = int(plan.text_encoder_device.split(":")[-1])
        vae_gpu = int(plan.vae_device.split(":")[-1])

        tokenizer = None
        scheduler = None
        text_encoder = None
        vae = None
        transformer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", **source_kwargs)
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", **source_kwargs)
            # Load to CPU first, then move to GPU to avoid caching_allocator_warmup
            # allocating an extra ~1 GiB peak on the target GPU during loading.
            text_encoder = AutoModel.from_pretrained(
                MODEL_ID,
                subfolder="text_encoder",
                **self._model_load_kwargs(dtype),
            )
            text_encoder = text_encoder.to(f"cuda:{text_encoder_gpu}")
            vae = AutoencoderKL.from_pretrained(
                MODEL_ID,
                subfolder="vae",
                **self._model_load_kwargs(dtype, device_map={"": vae_gpu}),
            )
            offload_folder = str(Path(__file__).parent / "offload_cache")
            os.makedirs(offload_folder, exist_ok=True)
            transformer = ZImageTransformer2DModel.from_pretrained(
                MODEL_ID,
                subfolder="transformer",
                device_map="balanced",
                offload_folder=offload_folder,
                **self._model_load_kwargs(dtype, max_memory=plan.transformer_max_memory),
            )
        except Exception:
            for _component in (text_encoder, vae, transformer):
                if _component is not None:
                    try:
                        _component.to("cpu")
                    except Exception:
                        pass
                    del _component
            del tokenizer, scheduler, text_encoder, vae, transformer
            gc.collect()
            for _gid in range(torch.cuda.device_count()):
                with torch.cuda.device(_gid):
                    torch.cuda.empty_cache()
            raise

        self.pipe = ZImagePipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        self.placement = ModelPlacement(
            mode=plan.mode,
            execution_device=plan.execution_device,
            dtype_name=self._dtype_name(dtype),
            text_encoder_device=plan.text_encoder_device,
            vae_device=plan.vae_device,
            transformer_device_map=dict(getattr(transformer, "hf_device_map", {}) or {}),
            transformer_max_memory=plan.transformer_max_memory,
            gpu_ids=plan.gpu_ids,
        )

        print(f"[Generator] Text encoder on {self.placement.text_encoder_device}")
        print(f"[Generator] VAE on {self.placement.vae_device}")
        print(f"[Generator] Transformer sharded across: {self.placement.transformer_device_map}")

    def _apply_memory_optimizations(self) -> None:
        """Use VRAM-friendly options before deciding to offload."""
        if self.pipe is None:
            return

        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            if hasattr(self.pipe.vae, "enable_slicing"):
                self.pipe.vae.enable_slicing()
                print("[Generator] Enabled VAE slicing")
            if hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()
                print("[Generator] Enabled VAE tiling")

        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[Generator] Enabled xFormers attention")
            except Exception:
                pass

    def _move_pipeline_to_gpu(self) -> None:
        """Try to keep the whole pipeline on GPU."""
        print(f"[Generator] Moving pipeline to {self.device}...")
        self.pipe.to(self.device)
        dtype = self._torch_dtype()
        self.placement = ModelPlacement(
            mode="single",
            execution_device=self.device,
            dtype_name=self._dtype_name(dtype),
            text_encoder_device=self.device,
            vae_device=self.device,
            gpu_ids=[0] if self.device == "cuda" else [],
        )
        print("[Generator] Pipeline fully loaded on GPU")

    def _best_single_gpu_device(self) -> str:
        """返回当前可用 GPU 中总显存最大的那张，用于 CPU offload 模式的执行设备。"""
        if not torch.cuda.is_available():
            return self.device
        gpu_ids = self._get_multi_gpu_ids() or ([0] if torch.cuda.device_count() > 0 else [])
        if not gpu_ids:
            return self.device
        best = max(gpu_ids, key=lambda gid: self._gpu_total_gib(gid))
        return f"cuda:{best}"

    def _enable_model_cpu_offload(self) -> None:
        """Prefer model-level offload over sequential offload for better GPU use."""
        exec_device = self._best_single_gpu_device()
        print(f"[Generator] Enabling model CPU offload on {exec_device}...")
        self.pipe.enable_model_cpu_offload(device=exec_device)
        print("[Generator] Model CPU offload enabled")

    def release_memory(self) -> None:
        """Aggressively free VRAM/RAM held by torch after a job or OOM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        print("[Generator] Released cached memory")
        
    def load_model(self) -> None:
        """加载模型"""
        if self.loaded:
            return
            
        from config import MODEL_REVISION
        revision = MODEL_REVISION.strip() if MODEL_REVISION else None
        rev_info = f" (revision: {revision})" if revision else " (latest)"
        dtype = self._torch_dtype()
        print(f"[Generator] Loading model: {MODEL_ID}{rev_info}")
        print(
            f"[Generator] Device: {self.device}, CPU Offload: {USE_CPU_OFFLOAD}, "
            f"Multi-GPU: {MULTI_GPU_MODE}, DType: {self._dtype_name(dtype)}"
        )

        if self._multi_gpu_enabled():
            try:
                print("[Generator] Trying multi-GPU sharded load...")
                self._load_model_multi_gpu()
                self._apply_memory_optimizations()
                self.loaded = True
                print("[Generator] Model loaded successfully with multi-GPU sharding!")
                return
            except Exception as e:
                print(f"[Generator] Multi-GPU load failed: {type(e).__name__}: {e}")
                self.pipe = None
                self.loaded = False
                self.placement = ModelPlacement(
                    execution_device=self.device,
                    dtype_name=self._dtype_name(dtype),
                )
                self.release_memory()
                _is_vram_issue = (
                    "out of memory" in str(e).lower()
                    or "unable to find a gpu placement" in str(e).lower()
                    or "enough vram" in str(e).lower()
                )
                if self._multi_gpu_required() and not _is_vram_issue:
                    raise
                if self._multi_gpu_required() and _is_vram_issue:
                    print(
                        "[Generator] MULTI_GPU_MODE=force but insufficient VRAM for multi-GPU plan; "
                        "falling back to CPU offload single-card mode..."
                    )
                else:
                    print("[Generator] Falling back to legacy single-device loading...")

        from diffusers import ZImagePipeline

        load_kwargs = self._model_load_kwargs(dtype)

        self.pipe = ZImagePipeline.from_pretrained(MODEL_ID, **load_kwargs)
        self._apply_memory_optimizations()

        if USE_CPU_OFFLOAD:
            self._enable_model_cpu_offload()
        else:
            try:
                self._move_pipeline_to_gpu()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(
                    "[Generator] CUDA OOM moving pipeline to GPU with USE_CPU_OFFLOAD=false; "
                    "falling back to CPU offload mode..."
                )
                self._enable_model_cpu_offload()
        if self.placement.text_encoder_device is None:
            self.placement = ModelPlacement(
                mode="single",
                execution_device=self.device,
                dtype_name=self._dtype_name(dtype),
                text_encoder_device=self.device,
                vae_device=self.device,
                gpu_ids=[0] if self.device == "cuda" else [],
            )

        self.loaded = True
        print("[Generator] Model loaded successfully!")
        
    def unload_model(self) -> None:
        """卸载模型释放显存"""
        if self.pipe is not None:
            pipe = self.pipe
            try:
                if hasattr(pipe, "reset_device_map"):
                    pipe.reset_device_map()
                elif hasattr(pipe, "remove_all_hooks"):
                    pipe.remove_all_hooks()
            except Exception as exc:
                print(f"[Generator] Failed to reset pipeline device map during unload: {exc}")

            components = getattr(pipe, "components", {}) or {}
            for name, component in components.items():
                if not isinstance(component, torch.nn.Module):
                    continue
                try:
                    component.to("cpu")
                except Exception as exc:
                    print(f"[Generator] Failed to move component `{name}` to CPU during unload: {exc}")

            del pipe
            self.pipe = None
            self.loaded = False
            self.placement = ModelPlacement(
                execution_device=self.device,
                dtype_name=self._dtype_name(torch.float32),
            )
            self.release_memory()
            print("[Generator] Model unloaded")

    def _get_execution_device(self) -> torch.device:
        """推理时创建随机数和中间张量所用设备。"""
        if self.pipe is not None:
            try:
                return torch.device(str(self.pipe._execution_device))
            except Exception:
                pass
            try:
                return torch.device(str(self.pipe.device))
            except Exception:
                pass

        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device(self.device)

    def _fallback_single_card_generate(
        self,
        params: GenerationParams,
        actual_seed: int,
        start_time: float,
    ) -> tuple[Image.Image, dict]:
        """NaN/Inf latents detected in multi-GPU path — reload single-card CPU-offload and retry."""
        if not USE_CPU_OFFLOAD:
            raise RuntimeError(
                "Multi-GPU generation produced invalid latents, and USE_CPU_OFFLOAD=false blocks CPU fallback"
            )

        from diffusers import ZImagePipeline

        print("[Generator] Reloading model in single-card CPU-offload mode for NaN-fallback re-run...")
        self.unload_model()
        dtype = self._torch_dtype()
        self.pipe = ZImagePipeline.from_pretrained(MODEL_ID, **self._model_load_kwargs(dtype))
        self._apply_memory_optimizations()
        self._enable_model_cpu_offload()
        if self.placement.text_encoder_device is None:
            self.placement = ModelPlacement(
                mode="single",
                execution_device=self.device,
                dtype_name=self._dtype_name(dtype),
                text_encoder_device=self.device,
                vae_device=self.device,
                gpu_ids=[0] if self.device == "cuda" else [],
            )
        self.loaded = True

        execution_device = self._get_execution_device()
        fb_generator = torch.Generator(device=str(execution_device)).manual_seed(actual_seed)
        result = self.pipe(
            prompt=params.prompt,
            height=params.height,
            width=params.width,
            num_inference_steps=params.steps,
            guidance_scale=0.0,
            generator=fb_generator,
        )
        image = result.images[0]
        elapsed = time.time() - start_time
        print(f"[Generator] NaN-fallback generated in {elapsed:.2f}s")
        return image, {
            "seed": actual_seed,
            "steps": params.steps,
            "width": params.width,
            "height": params.height,
            "elapsed_seconds": round(elapsed, 2),
            "model": MODEL_ID,
            "execution_device": str(execution_device),
            "placement_mode": self.placement.mode,
            "dtype": self.placement.dtype_name,
            "nan_fallback": True,
        }

    def _retry_multi_gpu_with_bf16(
        self,
        params: GenerationParams,
        actual_seed: int,
        start_time: float,
    ) -> Optional[tuple[Image.Image, dict]]:
        """Consumer RTX cards can produce NaNs in fp16 for this model; retry once in bf16."""
        if self._torch_dtype() == torch.bfloat16:
            return None
        if self.device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            return None

        previous_dtype = self._dtype_name(self._torch_dtype())
        print(
            f"[Generator] Invalid latents detected with {previous_dtype} multi-GPU inference. "
            "Reloading in bf16 and retrying before any CPU fallback..."
        )
        self._dtype_override = "bf16"
        self.unload_model()

        retry_params = replace(params, seed=actual_seed)
        image, metadata = self.generate(retry_params)
        metadata["precision_retry"] = "bf16"
        metadata["initial_invalid_dtype"] = previous_dtype
        metadata["elapsed_seconds"] = round(time.time() - start_time, 2)
        return image, metadata

    def generate(self, params: GenerationParams) -> tuple[Image.Image, dict]:
        """
        生成图像
        
        Returns:
            tuple: (生成的图像, 元数据dict)
        """
        params.validate()
        
        if not self.loaded:
            self.load_model()

        placement_mode_for_metadata = self.placement.mode
        dtype_for_metadata = self.placement.dtype_name
        transformer_device_map_for_metadata = dict(self.placement.transformer_device_map or {})

        # 设置随机种子
        actual_seed = params.seed if params.seed >= 0 else int(time.time() * 1000) % (2**32)
        execution_device = self._get_execution_device()
        generator = torch.Generator(device=str(execution_device)).manual_seed(actual_seed)

        print(
            f"[Generator] Generating: {params.width}x{params.height}, steps={params.steps}, "
            f"seed={actual_seed}, exec_device={execution_device}"
        )
        print(f"[Generator] Prompt: {params.prompt[:100]}...")
        
        start_time = time.time()

        if self.placement.mode == "multi_gpu":
            latent_result = self.pipe(
                prompt=params.prompt,
                height=params.height,
                width=params.width,
                num_inference_steps=params.steps,
                guidance_scale=0.0,
                generator=generator,
                output_type="latent",
            )

            latents = latent_result.images

            # --- NaN/Inf guard ---
            if not torch.isfinite(latents).all():
                nan_count = (~torch.isfinite(latents)).sum().item()
                print(
                    f"[Generator] WARNING: latents contain {nan_count} NaN/Inf values before VAE decode! "
                    "Trying safer recovery path..."
                )
                retry_result = self._retry_multi_gpu_with_bf16(params, actual_seed, start_time)
                if retry_result is not None:
                    return retry_result
                return self._fallback_single_card_generate(params, actual_seed, start_time)
            # ---------------------

            text_encoder_device = None
            destructive_decode_cleanup = False
            text_encoder_device = self._offload_text_encoder_for_vae_decode()
            moved_modules: list[tuple[str, torch.nn.Module, str]] = []
            decode_succeeded = False
            try:
                vae_device = self.placement.vae_device or str(execution_device)
                latents = latents.to(device=vae_device, dtype=self.pipe.vae.dtype)
                latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
                try:
                    with torch.inference_mode():
                        decoded = self.pipe.vae.decode(latents, return_dict=False)[0]
                except RuntimeError as decode_error:
                    error_text = str(decode_error).lower()
                    if "out of memory" not in error_text and "unable to find an engine" not in error_text:
                        raise
                    moved_modules = self._offload_transformer_for_vae_decode()
                    if not moved_modules:
                        destructive_decode_cleanup = self._discard_transformer_for_vae_decode() or destructive_decode_cleanup
                    with torch.inference_mode():
                        decoded = self.pipe.vae.decode(latents, return_dict=False)[0]

                image = self.pipe.image_processor.postprocess(decoded.detach(), output_type="pil")[0]
                decode_succeeded = True
            finally:
                if decode_succeeded:
                    if "decoded" in locals():
                        del decoded
                    del latents
                    gc.collect()
                    for gpu_id in range(torch.cuda.device_count()):
                        with torch.cuda.device(gpu_id):
                            torch.cuda.empty_cache()
                    if destructive_decode_cleanup:
                        print(
                            "[Generator] Freed decode-conflicting modules via meta discard; "
                            "unloading model after this job so the next job reloads cleanly"
                        )
                        self.unload_model()
                    else:
                        try:
                            self._restore_transformer_after_vae_decode(moved_modules)
                            self._restore_text_encoder_after_vae_decode(text_encoder_device)
                        except Exception as restore_error:
                            print(f"[Generator] Failed to restore modules after VAE decode: {restore_error}")
                            self.unload_model()
                else:
                    self.unload_model()
        else:
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
            "execution_device": str(execution_device),
            "placement_mode": placement_mode_for_metadata,
            "dtype": dtype_for_metadata,
        }

        if transformer_device_map_for_metadata:
            metadata["transformer_device_map"] = transformer_device_map_for_metadata

        return image, metadata

    def get_gpu_status(self) -> dict:
        """获取 GPU 状态（不包含 name，让 config.py 的 GPU_INFO.name 生效）"""
        if not torch.cuda.is_available():
            return {"available": False}

        devices = []
        for gpu_id in range(torch.cuda.device_count()):
            devices.append(
                {
                    "id": gpu_id,
                    "device_name": torch.cuda.get_device_name(gpu_id),
                    "memory_total_gb": round(torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3, 1),
                    "memory_used_gb": round(torch.cuda.memory_allocated(gpu_id) / 1024**3, 2),
                    "memory_cached_gb": round(torch.cuda.memory_reserved(gpu_id) / 1024**3, 2),
                }
            )

        aggregate_name = devices[0]["device_name"]
        if len(devices) > 1:
            aggregate_name = " + ".join(device["device_name"] for device in devices)

        return {
            "available": True,
            "device_name": aggregate_name,  # 原始设备名，不覆盖自定义 name
            "memory_total_gb": round(sum(device["memory_total_gb"] for device in devices), 1),
            "memory_used_gb": round(sum(device["memory_used_gb"] for device in devices), 2),
            "memory_cached_gb": round(sum(device["memory_cached_gb"] for device in devices), 2),
            "device_count": len(devices),
            "placement_mode": self.placement.mode,
            "devices": devices,
        }


# 全局单例
_generator: Optional[ZImageGenerator] = None
_generation_lock = threading.Lock()


def get_generator() -> ZImageGenerator:
    """获取生成器单例"""
    global _generator
    if _generator is None:
        _generator = ZImageGenerator()
    return _generator


def get_generation_lock() -> threading.Lock:
    """获取进程内共享生成锁，避免多个入口并发抢同一份模型。"""
    return _generation_lock


if __name__ == "__main__":
    # 测试
    gen = get_generator()
    gen.load_model()
    
    params = GenerationParams(
        prompt="a cute orange cat sleeping in warm sunlight, photorealistic",
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        seed=42,
    )
    
    image, meta = gen.generate(params)
    
    output_path = Path("../outputs/test_generator.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    print(f"Saved to: {output_path}")
    print(f"Metadata: {meta}")

from __future__ import annotations

import ctypes
import gc
import logging
import os
import time
from importlib.util import find_spec
from pathlib import Path
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable

import psutil
import torch
import torch.nn as nn

from config import PipelineConfig

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

# 获得当前模块独立的 Logger 实例 (依赖标准库层级冒泡)
logger: logging.Logger = logging.getLogger(__name__)


def lock_glibc_allocator() -> None:
    """在初始化阶段锁死 glibc ptmalloc 的动态阈值，强制 >128KB 的张量直接走 mmap/munmap"""
    try:
        libc = ctypes.CDLL("libc.so.6")
        M_TRIM_THRESHOLD = -1
        M_MMAP_THRESHOLD = -3
        M_ARENA_MAX = -8

        libc.mallopt(M_TRIM_THRESHOLD, 131072)  # 128 KB
        libc.mallopt(M_MMAP_THRESHOLD, 131072)  # 128 KB (禁止动态上调阈值)
        libc.mallopt(M_ARENA_MAX, 2)            # 限制多线程 Arena 数量
    except Exception:
        pass


def detach_mmap_tensors(pipe: DiffusionPipeline) -> None:
    """将管线内所有模型组件的权重从 safetensors 的 file-backed mmap 内存页
    剥离为标准匿名内存 (Anonymous RAM)，规避 Linux 6.18.57+ KFD userptr 死锁。
    """
    for module in pipe.components.values():
        if isinstance(module, nn.Module):
            for p in module.parameters():
                p.data = p.data.clone()
            for b in module.buffers():
                b.data = b.data.clone()


def vacuum_memory() -> None:
    """执行垃圾回收与显存缓存清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # 读取当前主进程的常驻内存 (RSS)
    process = psutil.Process(os.getpid())
    ram_rss_mb: float = process.memory_info().rss / (1024**2)

    if torch.cuda.is_available():
        # memory_allocated: 当前张量实际占用的 VRAM
        # memory_reserved: PyTorch 缓存分配器向 GPU 申请的总 VRAM
        vram_allocated_mb: float = torch.cuda.memory_allocated() / (1024**2)
        vram_reserved_mb: float = torch.cuda.memory_reserved() / (1024**2)
        logger.debug(
            f"显存回收完毕 | 进程 RAM: {ram_rss_mb:.0f} MB | "
            f"VRAM 占用: {vram_allocated_mb:.0f} MB (分配器预留: {vram_reserved_mb:.0f} MB)"
        )
    else:
        logger.debug(f"内存回收完毕 | 进程 RAM: {ram_rss_mb:.0f} MB")


def inject_complex_probe(pipe: DiffusionPipeline) -> Callable[[], None] | None:
    """挂载前向传播 Hook，检测并记录流经 pipe.transformer 的复数张量 (complex64/complex128)"""
    logger.info("正在为 transformer 挂载复数张量检测 Hook...")

    transformer: nn.Module | None = getattr(pipe, "transformer", None)
    if transformer is None:
        logger.error("未找到 pipe.transformer 实例")
        return None

    hook_handles: list[Any] = []
    captured_modules: set[str] = set()

    def scan_tensor(t: Any, module_info: str, io_type: str) -> None:
        """递归检查输入输出参数，定位复数张量"""
        if isinstance(t, torch.Tensor) and t.dtype in (torch.complex64, torch.complex128):
            if module_info not in captured_modules:
                captured_modules.add(module_info)
                logger.warning(
                    f"检测到复数张量 | 模块: {module_info} | "
                    f"位置: {io_type} | Shape: {tuple(t.shape)} | Dtype: {t.dtype}"
                )
        elif isinstance(t, (tuple, list)):
            for item in t:
                scan_tensor(item, module_info, io_type)
        elif isinstance(t, dict):
            for v in t.values():
                scan_tensor(v, module_info, io_type)

    def pre_hook(module: nn.Module, args: Any, kwargs: Any, module_name: str) -> None:
        scan_tensor(args, module_name, "Input Args")
        scan_tensor(kwargs, module_name, "Input Kwargs")

    def post_hook(module: nn.Module, args: Any, output: Any, module_name: str) -> None:
        scan_tensor(output, module_name, "Output Tensor")

    # 遍历子模块并注册 Hook (要求 PyTorch >= 2.0 以支持 with_kwargs)
    for name, module in transformer.named_modules():
        mod_info: str = f"{name} ({module.__class__.__name__})"

        h_pre = module.register_forward_pre_hook(
            lambda m, a, kw, info=mod_info: pre_hook(m, a, kw, info),
            with_kwargs=True,
        )
        h_post = module.register_forward_hook(
            lambda m, a, o, info=mod_info: post_hook(m, a, o, info)
        )
        hook_handles.append(h_pre)
        hook_handles.append(h_post)

    logger.info(f"已在 {len(hook_handles) // 2} 个子模块上注册检测 Hook")

    def remove_probe() -> None:
        """移除已注册的所有 Hook"""
        for h in hook_handles:
            h.remove()
        logger.debug("复数张量检测 Hook 已移除")

    return remove_probe

def resolve_attention_backend() -> str:
    """探测运行环境是否包含 flash_attn，若未安装则回退至 PyTorch 原生 _native_flash"""
    if find_spec("flash_attn") is not None:
        return "flash"
    logger.debug("未检测到 flash_attn，Attention 后端已自动回退至 _native_flash")
    return "_native_flash"


def calculate_vae_batch_size(
    num_samples: int,
    unit_cost_gb: float = 1.25,
    max_cap: int = 8,
) -> int:
    """根据运行时环境划定系统保留显存，基于 512-Tiling 单图 ~1.25GB 成本计算最优 Batch Size。

    :param num_samples: 待解码的潜变量总数量 (len(latents_buffer))
    :param unit_cost_gb: 512-Tiling 规格下单张图片分配器预留成本 (默认 1.25 GB)
    :param max_cap: 批次上限 (防止批次过大导致 CPU 异步落盘队列积压内存)
    :return: 规整到 2 的幂次的最优 batch_size (1, 2, 4, 8...)
    """
    if num_samples <= 1:
        return 1

    # 1. 探测硬件总显存与当前驱动层空闲显存
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb: float = free_bytes / (1024**3)
    total_gb: float = total_bytes / (1024**3)

    # 2. 划定系统保留 (Headroom Reserve)
    # WSL2 环境必须防范 WDDM 换页挤压 DWM；原生 Linux 仅需覆盖极低桌面底噪
    is_wsl: bool = os.path.exists("/dev/dxg")
    system_reserve_gb: float = 3.5 if is_wsl else 1.5

    # 3. 计入 PyTorch 缓存池内部已圈占但处于空闲的内存 (Pooled Free)
    pooled_free_gb: float = (
        torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    ) / (1024**3)

    # 4. 计算业务实际可用的安全显存净值
    usable_gb: float = max(
        0.0, (free_gb + pooled_free_gb) - system_reserve_gb
    )

    # 5. 计算理论批次大小：可用空间 // 单张成本 (1.25 GB)
    raw_bs: int = int(usable_gb // unit_cost_gb)

    # 6. 受上限与剩余样本数截断，并用位运算对齐到 2 的幂次 (1, 2, 4, 8...)
    target_bs: int = min(raw_bs, max_cap, num_samples)
    if target_bs <= 1:
        return 1

    # 取 <= target_bs 的最大 2 的幂次 (例: 7 -> 4, 4 -> 4, 3 -> 2)
    return 1 << (target_bs.bit_length() - 1)


def save_image_task(
    img: Image.Image,
    cfg: PipelineConfig,
    seed: int,
    key: str,
) -> None:
    """后台线程异步执行 PNG 压缩与落盘，统一提供目录检查与 I/O 异常保护"""
    if not cfg.outdir.is_dir():
        logger.error(
            f"输出目录不存在，跳过图片保存 (seed={seed}, key={key}): {cfg.outdir}"
        )
        return

    out_file: str = cfg.format_filename(seed, key)
    out_path: Path = cfg.outdir / out_file
    try:
        t0: float = time.time()
        img.save(out_path)
        logger.info(
            f"{out_file} 保存完成 | 异步I/O: {time.time() - t0:.2f}s"
        )
    except OSError as e:
        logger.error(f"{out_file} 保存失败: {e}")

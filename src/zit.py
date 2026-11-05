from __future__ import annotations

import cProfile
import io
import logging
import os
import pstats
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from base import BasePipeline
from config import CompileMode, ModelType, OffloadStrategy, PipelineConfig
from utils import (
    calculate_vae_batch_size,
    detach_mmap_tensors,
    inject_complex_probe,
    resolve_attention_backend,
    vacuum_memory,
)

if TYPE_CHECKING:
    from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput
    from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
    from PIL import Image

# 获得当前模块独立的 Logger 实例 (依赖标准库层级冒泡)
logger: logging.Logger = logging.getLogger(__name__)

# 性能诊断分析接口 (默认关闭，开启方式: DEBUG_PROFILE=1 python main.py)
ENABLE_PROFILING: bool = os.environ.get("DEBUG_PROFILE", "0") == "1"


def inject_real_rope_patch(pipe: ZImagePipeline) -> None:
    """将 Z-Image 管线的底层 RoPE 复数运算替换为纯实数运算，规避 torch.compile 图断裂"""
    logger.debug("正在替换底层 RoPE 为纯实数运算...")

    # 1. 局部延迟导入，避免在模块加载阶段触发 diffusers 初始化
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_z_image import RopeEmbedder

    # 2. 闭包内定义实数算子
    def precompute_freqs_cis_real(
        dim: list[int], end: list[int], theta: float = 256.0
    ) -> list[torch.Tensor]:
        with torch.device("cpu"):
            freqs_real: list[torch.Tensor] = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (
                    theta
                    ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
                )
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_real_i = torch.stack(
                    [torch.cos(freqs), torch.sin(freqs)], dim=-1
                ).flatten(-2, -1)
                freqs_real.append(freqs_real_i)
            return freqs_real

    class RealZSingleStreamAttnProcessor:
        _attention_backend = None
        _parallel_config = None

        def __call__(
            self,
            attn: Any,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            freqs_cis: torch.Tensor | None = None,
        ) -> torch.Tensor:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            def apply_rotary_emb_real(
                x_in: torch.Tensor, freqs_real: torch.Tensor
            ) -> torch.Tensor:
                with torch.amp.autocast("cuda", enabled=False):
                    x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
                    f = freqs_real.view(*freqs_real.shape[:-1], 1, -1, 2)

                    x_r, x_i = x.unbind(-1)
                    f_r, f_i = f.unbind(-1)

                    out_r = x_r * f_r - x_i * f_i
                    out_i = x_r * f_i + x_i * f_r

                    x_out = torch.stack([out_r, out_i], dim=-1).flatten(3)
                    return x_out.type_as(x_in)

            if freqs_cis is not None:
                query = apply_rotary_emb_real(query, freqs_cis)
                key = apply_rotary_emb_real(key, freqs_cis)

            dtype = query.dtype
            query, key = query.to(dtype), key.to(dtype)

            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

            hidden_states = hidden_states.flatten(2, 3).to(dtype)

            output = attn.to_out[0](hidden_states)
            if len(attn.to_out) > 1:
                output = attn.to_out[1](output)

            return output

    # 3. 替换 RopeEmbedder 静态方法与 Attention Processor 实例
    transformer = pipe.transformer
    RopeEmbedder.precompute_freqs_cis = staticmethod(precompute_freqs_cis_real)  # type: ignore

    if hasattr(transformer, "rope_embedder"):
        transformer.rope_embedder.freqs_cis = None

    patched_count: int = 0
    for _, module in transformer.named_modules():
        if (
            hasattr(module, "processor")
            and module.processor.__class__.__name__ == "ZSingleStreamAttnProcessor"
        ):
            module.processor = RealZSingleStreamAttnProcessor()
            patched_count += 1

    logger.debug(f"RoPE 实数算子替换完成，共更新 {patched_count} 个 Attention 节点")


class ZitPipeline(BasePipeline, model_type=ModelType.ZIT):
    """Z-Image-Turbo 推理管线实现"""

    def __init__(self) -> None:
        super().__init__()
        self.pipe: ZImagePipeline

    def init_pipeline(
        self,
        model_id: str,
        strategy: OffloadStrategy,
        compile_mode: CompileMode,
    ) -> None:
        """根据卸载策略加载模型权重并初始化管线"""
        print("=" * 55)
        print(" 初始化计算管线 (ZIT)...")
        print(f" 目标权重 : {model_id}")
        print(f" 调度策略 : {strategy.upper()}")
        print(f" 编译档位 : {compile_mode.upper()}")
        print("=" * 55)

        self.strategy = strategy
        start_time: float = time.time()

        from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

        logger.debug(f"环境初始化完成 | 耗时: {time.time() - start_time:.2f}s")

        try:
            t_load: float = time.time()
            pipe: ZImagePipeline = ZImagePipeline.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            # detach_mmap_tensors(pipe)

            attn_backend: str = resolve_attention_backend()
            pipe.transformer.set_attention_backend(attn_backend)

            # pipe.remove_hook = inject_complex_probe(pipe)
            inject_real_rope_patch(pipe)

            # 拆开显式设置 512 黄金切块属性与关闭 Slicing
            pipe.vae.enable_tiling()
            pipe.vae.disable_slicing()
            pipe.vae.tile_sample_min_size = 512
            pipe.vae.tile_latent_min_size = 64  # 512 // 8
            pipe.vae.tile_overlap_factor = 0.25

            logger.debug(
                f"权重加载与 Attention 后端挂载完成 | 耗时: {time.time() - t_load:.2f}s"
            )

            t_compile: float = time.time()
            if compile_mode == CompileMode.EXTREME:
                pipe.transformer = torch.compile(
                    pipe.transformer, mode="reduce-overhead", fullgraph=False
                )
            elif compile_mode == CompileMode.DEFAULT:
                pipe.transformer = torch.compile(
                    pipe.transformer, mode="default", fullgraph=False, dynamic=True
                )

            if compile_mode != CompileMode.NONE:
                logger.debug(
                    f"底层算子编译 ({compile_mode.name}) 挂载完成 | 耗时: {time.time() - t_compile:.2f}s"
                )

            t_strategy: float = time.time()
            if strategy == OffloadStrategy.NONE:
                pipe.to("cuda")
            elif strategy == OffloadStrategy.MODULE:
                pipe.enable_model_cpu_offload()
            elif strategy == OffloadStrategy.SEQ:
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.to("cpu")
            logger.debug(
                f"显存调度策略 ({strategy.name}) 配置完成 | 耗时: {time.time() - t_strategy:.2f}s"
            )

            self.pipe = pipe

            total_time: float = time.time() - start_time
            print("=" * 55)
            print(f" 管线初始化完毕 | 总耗时: {total_time:.2f}s")
            print("=" * 55 + "\n")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            sys.exit(1)

    def _extract_text_embeds(
        self, prompts_dict: dict[str, str]
    ) -> list[tuple[str, list[torch.Tensor]]]:
        """预计算文本特征并隔离至系统内存"""
        cpu_embeds_buffer: list[tuple[str, list[torch.Tensor]]] = []
        for key, text in prompts_dict.items():
            t_bake: float = time.time()
            with torch.no_grad():
                encoded = self.pipe._encode_prompt(prompt=text)
            cpu_embeds: list[torch.Tensor] = [
                t.to(device="cpu", copy=True) for t in encoded
            ]
            cpu_embeds_buffer.append((key, cpu_embeds))

            shapes: list[tuple[int, ...]] = [tuple(t.shape) for t in cpu_embeds]
            logger.debug(
                f"[{key}] 文本特征编码完成 | 尺寸: {shapes} | 耗时: {time.time() - t_bake:.2f}s"
            )
            del encoded

        return cpu_embeds_buffer

    def _warmup_pipeline(
        self,
        cpu_embeds_buffer: list[tuple[str, list[torch.Tensor]]],
        cfg: PipelineConfig,
    ) -> None:
        """执行单步空跑，触发底层算子编译与显存预分配"""
        for key, cpu_embeds in cpu_embeds_buffer:
            t_warm_key: float = time.time()
            cuda_embeds: list[torch.FloatTensor] = cast(
                list[torch.FloatTensor],
                [
                    t.to(device="cuda", dtype=self.pipe.transformer.dtype)
                    for t in cpu_embeds
                ],
            )
            _ = self.pipe(
                prompt_embeds=cuda_embeds,
                width=cfg.width,
                height=cfg.height,
                num_inference_steps=1,
                guidance_scale=0.0,
                output_type="latent",
            )

            logger.debug(
                f"[{key}] 单步预热完成 | 耗时: {time.time() - t_warm_key:.2f}s"
            )
            del cuda_embeds

    def _run_standard_pipeline(self, cfg: PipelineConfig) -> None:
        """执行标准推理: 依赖 Accelerate 原生机制进行显存调度"""
        logger.info("阶段一: 提取文本特征")

        t_start: float = time.time()
        cpu_embeds_buffer = self._extract_text_embeds(cfg.prompts)
        vacuum_memory()
        logger.info(f"文本特征提取完成 | 耗时: {time.time() - t_start:.2f}s")

        logger.info("阶段二: 启动推理循环")

        logger.info("启动管线预热与资源预分配...")
        t_warm: float = time.time()
        self._warmup_pipeline(cpu_embeds_buffer, cfg)
        vacuum_memory()
        logger.info(f"管线预热与资源预分配完成 | 耗时: {time.time() - t_warm:.2f}s")

        for i in range(cfg.count):
            current_seed: int = (
                cfg.seeds[i]
                if (isinstance(cfg.seeds, list) and i < len(cfg.seeds))
                else round(time.time() * 1000)
            )
            generator: torch.Generator = torch.Generator(device="cpu")
            logger.info(f"[迭代 {i+1}/{cfg.count}] 设定随机种子: {current_seed}")

            for key, cpu_embeds in cpu_embeds_buffer:
                t_infer: float = time.time()
                generator.manual_seed(current_seed)
                cuda_embeds: list[torch.FloatTensor] = cast(
                    list[torch.FloatTensor],
                    [
                        t.to(device="cuda", dtype=self.pipe.transformer.dtype)
                        for t in cpu_embeds
                    ],
                )

                from diffusers.pipelines.z_image.pipeline_output import (
                    ZImagePipelineOutput,
                )

                output = cast(
                    ZImagePipelineOutput,
                    self.pipe(
                        prompt_embeds=cuda_embeds,
                        width=cfg.width,
                        height=cfg.height,
                        num_inference_steps=(cfg.steps + 1),
                        guidance_scale=0.0,
                        generator=generator,
                    ),
                )
                image = output.images[0]

                out_file: str = cfg.format_filename(current_seed, key)
                image.save(cfg.outdir / out_file)
                logger.info(
                    f"{out_file} 保存完成 | 耗时: {time.time() - t_infer:.2f}s"
                )
                del cuda_embeds
                del output, image

        del cpu_embeds_buffer
        vacuum_memory()
        logger.info(
            f"标准管线执行完毕，显存已清理 | 总耗时: {time.time() - t_start:.2f}s"
        )

    def _run_text_offload_pipeline(self, cfg: PipelineConfig) -> None:
        """执行文本卸载推理: 仅卸载 Text Encoder，Transformer 与 VAE 常驻显存"""
        pipe = self.pipe
        logger.info("阶段一: 提取文本特征并隔离文本编码器")

        # 唤醒文本编码器
        if getattr(pipe, "text_encoder", None) is None:
            pipe.text_encoder = pipe.text_encoder_backup

        t_start: float = time.time()
        pipe.text_encoder.to("cuda")
        vacuum_memory()
        logger.debug(f"文本编码器加载至显存 | 耗时: {time.time() - t_start:.2f}s")

        cpu_embeds_buffer = self._extract_text_embeds(cfg.prompts)

        # 隔离文本编码器组件
        pipe.text_encoder_backup = pipe.text_encoder
        pipe.text_encoder.to("cpu")
        pipe.text_encoder = None
        vacuum_memory()
        logger.info(
            f"文本特征已转存，文本编码器已卸载至内存 | 耗时: {time.time() - t_start:.2f}s"
        )

        logger.info("阶段二: 启动推理循环 (文本卸载模式)")
        t_load: float = time.time()
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        vacuum_memory()
        logger.debug(
            f"Transformer 与 VAE 加载至显存 | 耗时: {time.time() - t_load:.2f}s"
        )

        logger.info("启动管线预热与资源预分配...")
        t_warm: float = time.time()
        self._warmup_pipeline(cpu_embeds_buffer, cfg)
        vacuum_memory()
        logger.info(f"管线预热与资源预分配完成 | 耗时: {time.time() - t_warm:.2f}s")

        for i in range(cfg.count):
            current_seed: int = (
                cfg.seeds[i]
                if (isinstance(cfg.seeds, list) and i < len(cfg.seeds))
                else round(time.time() * 1000)
            )
            generator: torch.Generator = torch.Generator(device="cpu")
            logger.info(f"[迭代 {i+1}/{cfg.count}] 设定随机种子: {current_seed}")

            for key, cpu_embeds in cpu_embeds_buffer:
                t_infer: float = time.time()
                generator.manual_seed(current_seed)
                cuda_embeds: list[torch.FloatTensor] = cast(
                    list[torch.FloatTensor],
                    [
                        t.to(device="cuda", dtype=pipe.transformer.dtype)
                        for t in cpu_embeds
                    ],
                )

                from diffusers.pipelines.z_image.pipeline_output import (
                    ZImagePipelineOutput,
                )

                output = cast(
                    ZImagePipelineOutput,
                    pipe(
                        prompt_embeds=cuda_embeds,
                        width=cfg.width,
                        height=cfg.height,
                        num_inference_steps=(cfg.steps + 1),
                        guidance_scale=0.0,
                        generator=generator,
                    ),
                )
                image = output.images[0]

                out_file: str = cfg.format_filename(current_seed, key)
                image.save(cfg.outdir / out_file)
                logger.info(
                    f"{out_file} 保存完成 | 耗时: {time.time() - t_infer:.2f}s"
                )
                del cuda_embeds
                del output, image

        pipe.transformer.to("cpu")
        pipe.vae.to("cpu")
        del cpu_embeds_buffer
        vacuum_memory()
        logger.info(
            f"文本卸载管线执行完毕，显存已清理 | 总耗时: {time.time() - t_start:.2f}s"
        )

    def _run_asym_offload_pipeline(self, cfg: PipelineConfig) -> None:
        """执行非对称卸载推理: 按计算阶段动态加载/卸载模型组件，通过内存缓冲潜变量"""
        pipe = self.pipe
        logger.info("阶段一: 提取文本特征并隔离文本编码器")

        # 唤醒文本编码器
        if getattr(pipe, "text_encoder", None) is None:
            pipe.text_encoder = pipe.text_encoder_backup

        t_start: float = time.time()
        pipe.text_encoder.to("cuda")
        vacuum_memory()
        logger.debug(f"文本编码器加载至显存 | 耗时: {time.time() - t_start:.2f}s")

        cpu_embeds_buffer = self._extract_text_embeds(cfg.prompts)

        # 隔离文本编码器组件
        pipe.text_encoder_backup = pipe.text_encoder
        pipe.text_encoder.to("cpu")
        pipe.text_encoder = None
        vacuum_memory()
        logger.info(
            f"文本特征已转存，文本编码器已卸载至内存 | 耗时: {time.time() - t_start:.2f}s"
        )

        logger.info("阶段二: 启动 Transformer 推理 (潜变量缓冲模式)")
        t_dit_load: float = time.time()
        pipe.transformer.to("cuda")
        vacuum_memory()
        logger.debug(f"Transformer 加载至显存 | 耗时: {time.time() - t_dit_load:.2f}s")

        logger.info("启动管线预热与资源预分配...")
        t_warm: float = time.time()
        self._warmup_pipeline(cpu_embeds_buffer, cfg)
        vacuum_memory()
        logger.info(f"管线预热与资源预分配完成 | 耗时: {time.time() - t_warm:.2f}s")

        latents_buffer: list[tuple[int, str, torch.Tensor]] = []
        for i in range(cfg.count):
            current_seed: int = (
                cfg.seeds[i]
                if (isinstance(cfg.seeds, list) and i < len(cfg.seeds))
                else round(time.time() * 1000)
            )
            generator: torch.Generator = torch.Generator(device="cpu")
            logger.info(f"[迭代 {i+1}/{cfg.count}] 设定随机种子: {current_seed}")

            for key, cpu_embeds in cpu_embeds_buffer:
                t_infer: float = time.time()
                generator.manual_seed(current_seed)
                cuda_embeds: list[torch.FloatTensor] = cast(
                    list[torch.FloatTensor],
                    [
                        t.to(device="cuda", dtype=pipe.transformer.dtype)
                        for t in cpu_embeds
                    ],
                )

                pr = None
                if ENABLE_PROFILING:
                    # 启动性能分析
                    pr = cProfile.Profile()
                    pr.enable()

                from diffusers.pipelines.z_image.pipeline_output import (
                    ZImagePipelineOutput,
                )

                output = cast(
                    ZImagePipelineOutput,
                    pipe(
                        prompt_embeds=cuda_embeds,
                        width=cfg.width,
                        height=cfg.height,
                        num_inference_steps=(cfg.steps + 1),
                        guidance_scale=0.0,
                        generator=generator,
                        output_type="latent",
                    ),
                )

                if pr is not None:
                    # 停止并输出性能分析
                    torch.cuda.synchronize()
                    pr.disable()

                    # 格式化输出耗时最长的前 15 个内部函数
                    s = io.StringIO()
                    sortby = "cumulative"
                    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    ps.print_stats(15)
                    logger.debug(f"\n=== Pipeline 性能分析 ===\n{s.getvalue()}")

                raw_latents: torch.Tensor = cast(torch.Tensor, output.images)
                latents_buffer.append(
                    (current_seed, key, raw_latents.to(device="cpu", copy=True))
                )
                logger.debug(
                    f"[{key}] 潜变量推理与内存缓冲完成 | 耗时: {time.time() - t_infer:.2f}s"
                )
                del cuda_embeds
                del output, raw_latents

        logger.info("阶段三: 卸载 Transformer 并加载 VAE")
        t_unload: float = time.time()
        pipe.transformer.to("cpu")
        del cpu_embeds_buffer
        vacuum_memory()
        logger.debug(f"Transformer 卸载至内存 | 耗时: {time.time() - t_unload:.2f}s")

        t_vae_load: float = time.time()
        pipe.vae.to("cuda")
        vacuum_memory()
        logger.debug(f"VAE 加载至显存 | 耗时: {time.time() - t_vae_load:.2f}s")

        logger.info("阶段四: VAE 批量解码与保存")

        save_futures: list[Future[None]] = []
        total_samples: int = len(latents_buffer)

        def _save_image_task(
            img: Image.Image, out_path: Path, out_file: str, dec_cost: float
        ) -> None:
            """后台线程异步执行 PNG 压缩与落盘 (Pillow 底层 libpng 自动释放 GIL)"""
            t0 = time.time()
            img.save(out_path)
            logger.info(
                f"{out_file} 保存完成 | GPU均摊解码: {dec_cost:.2f}s | 异步I/O: {time.time() - t0:.2f}s"
            )

        def _decode_latent_batch(
            batch_items: list[tuple[int, str, torch.Tensor]],
        ) -> list[Image.Image]:
            """执行单个微批次的 GPU 解码并收敛返回类型为 list[Image.Image]"""
            batch_latents = torch.cat([item[2] for item in batch_items], dim=0).to(
                device="cuda", dtype=pipe.vae.dtype, non_blocking=True
            )
            batch_latents = (
                batch_latents / pipe.vae.config.scaling_factor
            ) + pipe.vae.config.shift_factor

            decoded_tensors = pipe.vae.decode(batch_latents, return_dict=False)[0]
            # 使用 cast 在源头收敛联合类型
            # 用字符串字面量作为 cast 目标类型
            # Pylance 完美解析，运行时零求值报错
            images = cast(
                "list[Image.Image]",
                pipe.image_processor.postprocess(decoded_tensors, output_type="pil"),
            )
            del batch_latents, decoded_tensors
            torch.cuda.synchronize()
            return images

        # 取 (CPU逻辑核心数, total_samples) 的较小值
        io_workers: int = min(os.cpu_count() or 1, total_samples)
        # 计算并行解码数（经验公式： 1.25GB / 图）
        vae_batch_size: int = calculate_vae_batch_size(total_samples)
        logger.info(f"VAE解码并行度: batch_size={vae_batch_size}")

        with ThreadPoolExecutor(max_workers=io_workers) as io_pool, torch.inference_mode():
            for i in range(0, total_samples, vae_batch_size):
                t_dec = time.time()
                batch_items = latents_buffer[i : i + vae_batch_size]

                images = _decode_latent_batch(batch_items)
                avg_cost = (time.time() - t_dec) / len(images)

                for (current_seed, key, _), img in zip(batch_items, images):
                    out_file: str = cfg.format_filename(current_seed, key)
                    save_futures.append(
                        io_pool.submit(
                            _save_image_task,
                            img,
                            cfg.outdir / out_file,
                            out_file,
                            avg_cost,
                        )
                    )

        # 统一等待所有后台落盘完成
        for fut in save_futures:
            fut.result()

        pipe.vae.to("cpu")
        del latents_buffer
        # pipe.remove_hook()
        vacuum_memory()
        logger.info(
            f"非对称卸载管线执行完毕，显存已清理 | 总耗时: {time.time() - t_start:.2f}s"
        )

    def dispatch_pipeline(self, cfg: PipelineConfig) -> None:
        """任务分发: 根据配置策略分发推理任务"""
        if not isinstance(cfg.prompts, dict) or not cfg.prompts:
            raise ValueError("prompts 参数必须为非空字典")

        # 1. 出图前：写入展开后的初始配置，并重置内部种子收集器
        dump_path: Path | None = cfg.dump_toml(is_final=False)
        if dump_path is not None:
            logger.debug(f"初始配置已落盘: {dump_path}")

        try:
            if self.strategy in (
                OffloadStrategy.NONE,
                OffloadStrategy.MODULE,
                OffloadStrategy.SEQ,
            ):
                self._run_standard_pipeline(cfg)
            elif self.strategy == OffloadStrategy.TEXT:
                self._run_text_offload_pipeline(cfg)
            elif self.strategy == OffloadStrategy.ASYM:
                self._run_asym_offload_pipeline(cfg)
        finally:
            # 2. 出图后（或被中断时）：若已捕获到实际执行的种子，则回填更新配置文件
            if cfg._executed_seeds:
                dump_path: Path | None = cfg.dump_toml(is_final=True)
                if dump_path is not None:
                    logger.info(f"运行配置已归档至: {dump_path}")

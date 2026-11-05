from __future__ import annotations

import logging
import sys
from PIL import Image
from typing import Any

import zit  # noqa: F401  # 触发 ZitPipeline 子类注册
from base import BasePipeline
from config import (
    CompileMode,
    ModelType,
    OffloadStrategy,
    PipelineConfig,
    init_config,
    load_config,
    prompt_runtime_env,
)
from logger import setup_pipeline_logger
from utils import lock_glibc_allocator, vacuum_memory

# 获得当前模块独立的 Logger 实例
logger: logging.Logger = logging.getLogger(__name__)


def display_runtime_status(
    is_interactive: bool,
    cfg_name: str,
    model_type: ModelType,
    strategy: OffloadStrategy,
    compile_mode: CompileMode,
) -> None:
    """展示系统当前运行模式与基础环境配置"""
    print("=" * 55)
    mode_str = (
        "交互式终端 (Ctrl-D/'q'/'quit'/'exit' 退出, 生成中按 Ctrl+C 中断)"
        if is_interactive
        else "CLI 批处理"
    )
    print(f" 运行模式 : {mode_str}")
    print(f" 配置文件 : {cfg_name if cfg_name else '未指定'}")
    print(f" 模型类型 : {model_type.upper()}")
    print(f" 卸载策略 : {strategy.upper()}")
    print(f" 编译档位 : {compile_mode.upper()}")
    print("=" * 55 + "\n")


def refresh_pipeline(
    pipe: BasePipeline | None,
    current_state: dict[str, Any],
    target_state: dict[str, Any],
) -> tuple[BasePipeline, dict[str, Any]]:
    """
    统一的管线生命周期管理器:
    对比当前状态与目标状态，自动执行按需销毁与延迟加载，返回就绪的管线实例与最新状态
    """
    # 1. 状态差分与旧实例销毁
    if pipe is not None and current_state != target_state:
        logger.info("检测到管线配置变更:")
        for key, new_val in target_state.items():
            old_val = current_state.get(key)
            if old_val != new_val:
                old_str = (
                    getattr(old_val, "name", str(old_val))
                    if old_val is not None
                    else "None"
                )
                new_str = (
                    getattr(new_val, "name", str(new_val))
                    if new_val is not None
                    else "None"
                )
                logger.info(f"  - {key}: {old_str} -> {new_str}")

        logger.info("销毁旧管线实例并回收显存")
        del pipe
        pipe = None
        vacuum_memory()
    elif pipe is not None and current_state == target_state:
        # 命中缓存复用
        logger.info("管线配置未变更，直接复用当前管线实例")

    # 2. 延迟加载与状态覆写
    if pipe is None:
        pipe = BasePipeline.create(target_state["model_type"])
        pipe.init_pipeline(
            str(target_state["model_id"]),
            target_state["strategy"],
            target_state["compile_mode"],
        )
        current_state = target_state.copy()

    return pipe, current_state


def main() -> None:
    # 顶层入口注入通用业务标识 "T2I"，完成系统 Handler 挂载
    setup_pipeline_logger("T2I")

    # 锁定 glibc allocator
    lock_glibc_allocator()

    # 预加载 Pillow 核心插件，杜绝后台多线程并发保存时的插件导入竞争
    Image.preinit()

    init_ok, is_interactive, strategy, compile_mode, cfg_name, cfg = init_config()
    if not init_ok or cfg is None:
        sys.exit(1)

    # 初始状态为空，将实例化推迟到首次 dispatch 时进行
    pipe: BasePipeline | None = None

    if not is_interactive:
        # ---------------- CLI 批处理模式 ----------------
        display_runtime_status(
            is_interactive, cfg_name, cfg.model_type, strategy, compile_mode
        )

        if not cfg.prepare_outdir():
            sys.exit(1)

        if pipe is None:
            pipe = BasePipeline.create(cfg.model_type)
            pipe.init_pipeline(cfg.model_id, strategy, compile_mode)

        pipe.dispatch_pipeline(cfg)
        logger.info("批处理任务完成，系统安全退出")
        sys.exit(0)
    else:
        # ---------------- 交互式推理终端 ----------------
        display_runtime_status(
            is_interactive, cfg_name, cfg.model_type, strategy, compile_mode
        )

        cur_cfg_name: str = cfg_name
        current_state: dict[str, Any] = {}  # 运行时状态指示器

        while True:
            try:
                print("-" * 55)
                # 规避 TTY 宽字符退格计算错误，采用换行空白输入
                print(
                    f"输入指令 ['r': 运行 '{cur_cfg_name}', 'i': 参数, 'e': 环境, 或直接输入新的配置路径]:"
                )
                user_choice: str = input().strip()

                if not user_choice:
                    continue

                if user_choice.lower() in ("q", "quit", "exit"):
                    logger.info("收到退出指令，系统安全退出")
                    sys.exit(0)

                # --- 分支 0: 环境设置模式 ---
                if user_choice.lower() == "e":
                    strategy, compile_mode = prompt_runtime_env(
                        cfg, strategy, compile_mode
                    )
                    # 更新后刷新状态 UI，返回主循环等待派发
                    display_runtime_status(
                        is_interactive,
                        cur_cfg_name,
                        cfg.model_type,
                        strategy,
                        compile_mode,
                    )
                    continue

                # --- 分支 1: 手工微调模式 ---
                elif user_choice.lower() == "i":
                    if not cfg.prompt_generation_params():
                        continue

                    if not cfg.prepare_outdir():
                        continue

                    # 统一构建目标状态与管线调度
                    target_state = {
                        "model_type": cfg.model_type,
                        "model_id": cfg.model_id,
                        "strategy": strategy,
                        "compile_mode": compile_mode,
                    }
                    pipe, current_state = refresh_pipeline(
                        pipe, current_state, target_state
                    )

                    pipe.dispatch_pipeline(cfg)

                # --- 分支 2: 配置文件复用模式 ---
                else:
                    if user_choice.lower() != "r":
                        cur_cfg_name = user_choice

                    loaded_cfg, loaded_ok = load_config(cur_cfg_name)
                    if not loaded_ok:
                        continue

                    cfg = loaded_cfg

                    if not cfg.prepare_outdir():
                        continue

                    target_state = {
                        "model_type": cfg.model_type,
                        "model_id": cfg.model_id,
                        "strategy": strategy,
                        "compile_mode": compile_mode,
                    }

                    pipe, current_state = refresh_pipeline(
                        pipe, current_state, target_state
                    )

                    pipe.dispatch_pipeline(cfg)

            except EOFError:
                # 响应 Unix 终端原生的 Ctrl-D 终止符
                print()
                logger.info("收到 EOF 信号，系统安全退出")
                sys.exit(0)

            except KeyboardInterrupt:
                print()
                logger.warning("捕捉到中断信号，当前任务已终止")

                if pipe is not None:
                    logger.info("销毁当前管线实例以清理显存碎片")
                    del pipe
                    pipe = None

                current_state = {}
                vacuum_memory()
                continue


if __name__ == "__main__":
    main()

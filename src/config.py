import argparse
import itertools
import logging
import re
import tomllib
import tomli_w
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# 获得当前模块独立的 Logger 实例 (依赖标准库层级冒泡)
logger: logging.Logger = logging.getLogger(__name__)


# ==========================================
# 2. 全局配置与参数解析
# ==========================================
class ModelType(StrEnum):
    """模型架构类型"""

    ZIT = "zit"  # Z-Image-Turbo
    K2T = "k2t"  # Krea-2-Turbo


class OffloadStrategy(StrEnum):
    """硬件显存调度策略"""

    NONE = "none"  # 全量加载至显存
    MODULE = "module"  # Accelerate 模型级卸载
    SEQ = "seq"  # Accelerate 序列化卸载
    TEXT = "text"  # 文本编码器独立卸载
    ASYM = "asym"  # 非对称卸载的潜变量缓冲调度


class CompileMode(StrEnum):
    """底层 C++ 静态图编译档位"""

    NONE = "none"  # 原生 Eager 模式
    DEFAULT = "default"  # 基础算子融合 (兼容动态输入)
    EXTREME = "extreme"  # 极速模式 (降低额外开销)


DEFAULT_MODEL_TYPE: ModelType = ModelType.ZIT
DEFAULT_OFFLOAD_STRATEGY: OffloadStrategy = OffloadStrategy.ASYM
DEFAULT_COMPILE_MODE: CompileMode = CompileMode.DEFAULT
default_cfg_file: str = "default.toml"

# 匹配花括号参数块，如 {A, B, C} 或 {1..5}
_BRACE_PATTERN: re.Pattern[str] = re.compile(r"\{([^{}]+)\}")


def _parse_brace_token(token: str) -> list[str]:
    """解析单个花括号内的语法，支持 .. 连续范围与逗号枚举"""
    token_str: str = token.strip()
    if ".." in token_str:
        parts: list[str] = [x.strip() for x in token_str.split("..", 1)]
        if len(parts) == 2:
            start, end = parts[0], parts[1]
            # 1. 整数范围展开 (支持正序/逆序，如 1..3 或 3..1)
            if start.isdigit() and end.isdigit():
                s_idx, e_idx = int(start), int(end)
                step: int = 1 if s_idx <= e_idx else -1
                return [str(i) for i in range(s_idx, e_idx + step, step)]
            # 2. 单字符 ASCII 字母范围展开 (如 A..C 或 c..a)
            if len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
                s_code, e_code = ord(start), ord(end)
                step = 1 if s_code <= e_code else -1
                return [chr(i) for i in range(s_code, e_code + step, step)]

    return [x.strip() for x in token_str.split(",") if x.strip()]


def _expand_prompts(raw_prompts: dict[str, str]) -> dict[str, str]:
    """将含花括号语法的提示词字典按笛卡尔积展开为扁平字典"""
    expanded: dict[str, str] = {}

    for base_key, template in raw_prompts.items():
        matches: list[re.Match[str]] = list(_BRACE_PATTERN.finditer(template))
        if not matches:
            expanded[base_key] = template
            continue

        param_lists: list[list[str]] = [_parse_brace_token(m.group(1)) for m in matches]
        if any(not plist for plist in param_lists):
            raise ValueError(f"提示词 [{base_key}] 包含空的花括号参数块: {template}")

        num_params: int = len(param_lists)
        key_placeholders: int = base_key.count("{}")

        if key_placeholders > num_params:
            raise ValueError(
                f"键名 [{base_key}] 的占位符 '{{}}' 数量 ({key_placeholders}) "
                f"超出了提示词中的参数维度 ({num_params})"
            )

        # 将 Value 中的 {xxx} 替换为标准 {} 以便后续填充
        clean_template: str = _BRACE_PATTERN.sub("{}", template)

        for combo in itertools.product(*param_lists):
            # 清洗特殊字符生成安全的文件名片段 (空串兜底为 'empty')
            safe_tokens: list[str] = [
                re.sub(r"\W+", "", c) or "empty" for c in combo
            ]

            # 满秩与欠秩统一路由：前 K 个填入 Key 的 {}，剩余 N-K 个追加在末尾
            if key_placeholders > 0:
                head_part: str = base_key.format(*safe_tokens[:key_placeholders])
                tail_tokens: list[str] = safe_tokens[key_placeholders:]
                new_key: str = (
                    f"{head_part}_{'_'.join(tail_tokens)}" if tail_tokens else head_part
                )
            else:
                new_key = f"{base_key}_{'_'.join(safe_tokens)}"

            expanded[new_key] = clean_template.format(*combo)

    return expanded


@dataclass(slots=True)
class PipelineConfig:
    model_type: ModelType = DEFAULT_MODEL_TYPE
    model_id: str = "/your_path_to/zit-model"
    width: int = 1024
    height: int = 1024
    count: int = 1
    seeds: list[int] = field(default_factory=list)
    steps: int = 8
    outdir: Path = field(default_factory=lambda: Path("./outputs"))
    prefix: str = "zit"
    prompts: dict[str, str] = field(
        default_factory=lambda: {"base": "The quick brown fox jumps over the lazy dog."}
    )
    # 内部运行态记录：通过命名回调收集实际执行的随机种子
    _executed_seeds: list[int] = field(default_factory=list, init=False, repr=False)

    @classmethod
    def default(cls) -> "PipelineConfig":
        return cls()

    def to_toml_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "model_type": str(self.model_type),
                "model_id": self.model_id,
            },
            "generation": {
                "width": self.width,
                "height": self.height,
                "count": self.count,
                "seeds": self.seeds,
                "steps": self.steps,
                "outdir": str(self.outdir),
                "prefix": self.prefix,
            },
            "prompts": self.prompts,
        }

    @classmethod
    def from_toml_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        default_inst: PipelineConfig = cls.default()
        gen_data: dict[str, Any] = data.get("generation") or {}
        model_data: dict[str, Any] = data.get("model") or {}

        raw_type: Any = model_data.get("model_type")
        if raw_type is not None:
            try:
                model_type = ModelType(str(raw_type).lower())
            except ValueError:
                logger.warning(
                    f"未知的模型类型 '{raw_type}'，已回退至默认类型: {default_inst.model_type.name}"
                )
                model_type = default_inst.model_type
        else:
            model_type = default_inst.model_type

        outdir_val: Any = gen_data.get("outdir")
        outdir_path: Path = (
            Path(outdir_val)
            if isinstance(outdir_val, (str, Path))
            else default_inst.outdir
        )

        seeds_val: Any = gen_data.get("seeds")
        seeds_list: list[int] = (
            [int(s) for s in seeds_val]
            if isinstance(seeds_val, list)
            else default_inst.seeds
        )

        prompts_val: Any = data.get("prompts")
        clean_prompts: dict[str, str] = (
            {str(k): str(v) for k, v in prompts_val.items()}
            if isinstance(prompts_val, dict)
            else default_inst.prompts
        )

        return cls(
            model_type=model_type,
            model_id=str(model_data.get("model_id") or default_inst.model_id),
            width=int(gen_data.get("width") or default_inst.width),
            height=int(gen_data.get("height") or default_inst.height),
            count=int(gen_data.get("count") or default_inst.count),
            seeds=seeds_list,
            steps=int(gen_data.get("steps") or default_inst.steps),
            outdir=outdir_path,
            prefix=str(gen_data.get("prefix") or default_inst.prefix),
            prompts=_expand_prompts(clean_prompts),
        )

    def format_filename(self, seed: int, key: str) -> str:
        """生成输出图片文件名，并顺带捕获实际执行的随机种子"""
        if seed not in self._executed_seeds:
            self._executed_seeds.append(seed)
        return f"{self.prefix}-{seed}-{key}.png"

    def dump_toml(self, is_final: bool = False) -> Path:
        """
        将当前配置（含展开后的 prompts）落盘至 outdir/{prefix}.toml。
        - is_final=False: 初始落盘，清空历史执行记录，原样写入配置。
        - is_final=True: 结束落盘，注释原始 seeds 块（兼容多行），并写入实际执行的 seeds。
        """
        if not is_final:
            self._executed_seeds.clear()

        dump_path: Path = self.outdir / f"{self.prefix}.toml"
        toml_str: str = tomli_w.dumps(self.to_toml_dict())

        if is_final and self._executed_seeds:
            # 交由 tomli_w 独立渲染 seeds 键值对，确保单行/多行格式与全文一致
            orig_block: str = tomli_w.dumps({"seeds": self.seeds}).strip()
            actual_block: str = tomli_w.dumps({"seeds": self._executed_seeds}).strip()

            # 对原始 seeds 块的每一行统一追加 '# ' 注释前缀
            commented_orig: str = "\n".join(
                f"# {line}" for line in orig_block.splitlines()
            )
            toml_str = toml_str.replace(
                orig_block, f"{commented_orig}\n{actual_block}", 1
            )

        dump_path.write_text(toml_str, encoding="utf-8")
        return dump_path

    def prompt_generation_params(self) -> bool:
        """交互式采集并更新生成参数，返回是否成功完成设置"""
        print("交互设置: 请输入生成参数 (直接回车保持默认): ")
        try:
            print(f"图片宽度 (默认 {self.width}):")
            w = input().strip()
            self.width = int(w) if w else self.width

            print(f"图片高度 (默认 {self.height}):")
            h = input().strip()
            self.height = int(h) if h else self.height

            prompt = list(self.prompts.values())[0] if self.prompts else ""
            if prompt:
                print(f"当前提示词: {prompt[:50]}...")
                print("输入提示词 (直接回车复用):")
                p = input().strip()
                prompt = p if p else prompt
            else:
                print("输入提示词 (必填):")
                prompt = input().strip()
                if not prompt:
                    logger.warning("提示词不可为空，已退回主菜单")
                    return False
            self.prompts = _expand_prompts({"manual": prompt})

            print(f"生成数量 (默认 {self.count}):")
            c = input().strip()
            self.count = int(c) if c else self.count

            print("随机种子 (多种子用逗号分隔，留空则随机):")
            s_input = input().strip()
            self.seeds = (
                [int(s.strip()) for s in s_input.split(",")]
                if s_input
                else []
            )

            print(f"推理步数 (默认 {self.steps}):")
            st = input().strip()
            self.steps = int(st) if st else self.steps

            print(f"输出目录 (默认 {self.outdir}):")
            out = input().strip()
            self.outdir = Path(out) if out else self.outdir

            print(f"文件前缀 (默认 {self.prefix}):")
            pre = input().strip()
            self.prefix = pre if pre else self.prefix

            return True
        except ValueError:
            logger.warning("参数格式错误 (需要输入整数)，本次设置已中断")
            return False


def prompt_runtime_env(
    cfg: PipelineConfig,
    strategy: OffloadStrategy,
    compile_mode: CompileMode,
) -> tuple[OffloadStrategy, CompileMode]:
    """交互式更新运行环境参数 (模型类型、模型路径、卸载策略、编译档位)"""
    print("交互设置: 运行环境 (直接回车保持当前状态)")
    print(
        "[注意] 模型类型与路径允许手动指定，但始终会被最新读取的配置文件覆盖"
    )

    type_options = "/".join(t.value for t in ModelType)
    print(f"模型类型 (当前 {cfg.model_type.name}, 可选 {type_options}):")
    new_type = input().strip().lower()
    if new_type:
        try:
            cfg.model_type = ModelType(new_type)
        except ValueError:
            logger.warning(
                f"已忽略无效模型类型 '{new_type}'，保持当前参数: {cfg.model_type.name}"
            )

    current_model = cfg.model_id if cfg.model_id else "未指定"
    print(f"模型路径 (当前 {current_model}):")
    new_model = input().strip()
    if new_model:
        cfg.model_id = new_model

    strat_options = "/".join(s.value for s in OffloadStrategy)
    print(f"卸载策略 (当前 {strategy.name}, 可选 {strat_options}):")
    new_strat = input().strip().lower()
    if new_strat:
        try:
            strategy = OffloadStrategy(new_strat)
        except ValueError:
            logger.warning(
                f"已忽略无效策略 '{new_strat}'，保持当前参数: {strategy.name}"
            )

    comp_options = "/".join(c.value for c in CompileMode)
    print(f"编译档位 (当前 {compile_mode.name}, 可选 {comp_options}):")
    new_comp = input().strip().lower()
    if new_comp:
        try:
            compile_mode = CompileMode(new_comp)
        except ValueError:
            logger.warning(
                f"已忽略无效档位 '{new_comp}'，保持当前参数: {compile_mode.name}"
            )

    return strategy, compile_mode


def load_config(toml_path: str = default_cfg_file) -> tuple[PipelineConfig, bool]:
    """读取 TOML 配置文件，若文件缺失则回退并写入缺省参数"""
    default_config: PipelineConfig = PipelineConfig.default()
    try:
        with open(toml_path, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
    except FileNotFoundError:
        toml_string: str = tomli_w.dumps(default_config.to_toml_dict())
        logger.warning(
            f"配置文件 {toml_path} 缺失，已回退至缺省参数: \n\n{toml_string}"
        )
        return default_config, False
    except Exception as e:
        logger.error(f"配置文件解析失败: {e}")
        return default_config, False

    cfg: PipelineConfig = PipelineConfig.from_toml_dict(data)
    return cfg, True


def init_config() -> (
    tuple[bool, bool, OffloadStrategy, CompileMode, str, PipelineConfig]
):
    """解析命令行参数与显存调度策略，初始化执行状态"""
    parser = argparse.ArgumentParser(description="T2I 本地推理引擎")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="输入配置文件路径以执行单次批处理 (留空则进入交互模式)",
    )

    # 利用 StrEnum 在 CLI 边界自动收敛类型并拦截非法输入
    parser.add_argument(
        "-o",
        "--offload",
        type=OffloadStrategy,
        choices=list(OffloadStrategy),
        default=DEFAULT_OFFLOAD_STRATEGY,
        help="硬件显存调度策略",
    )
    parser.add_argument(
        "--compile",
        type=CompileMode,
        choices=list(CompileMode),
        default=DEFAULT_COMPILE_MODE,
        help="底层算子编译档位",
    )
    args: argparse.Namespace = parser.parse_args()

    init_ok: bool = True
    is_interactive: bool = args.config is None
    cfg_name: str = ""
    cfg: PipelineConfig = PipelineConfig.default()

    if args.config:
        cfg_name = args.config
        cfg, loaded_ok = load_config(args.config)
        if not loaded_ok:
            logger.error(f"配置文件加载失败: {cfg_name}")
            init_ok = False
        elif not cfg.model_id:
            logger.error("配置文件缺少核心参数: model_id")
            init_ok = False
    else:
        cfg, loaded_ok = load_config(default_cfg_file)
        cfg_name = default_cfg_file if loaded_ok else ""
        if not cfg.model_id:
            logger.error("内部缺省配置缺少核心参数: model_id")
            init_ok = False

    return init_ok, is_interactive, args.offload, args.compile, cfg_name, cfg

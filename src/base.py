from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from config import CompileMode, ModelType, OffloadStrategy, PipelineConfig

# 获得当前模块独立的 Logger 实例 (依赖标准库层级冒泡)
logger: logging.Logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """T2I 推理管线抽象基类与子类注册表"""

    _REGISTRY: ClassVar[dict[ModelType, type[BasePipeline]]] = {}

    def __init_subclass__(
        cls, model_type: ModelType | None = None, **kwargs: Any
    ) -> None:
        """在子类定义时自动将其注册到对应的 ModelType 枚举键下"""
        super().__init_subclass__(**kwargs)
        if model_type is not None:
            cls._REGISTRY[model_type] = cls

    @classmethod
    def create(cls, model_type: ModelType) -> BasePipeline:
        """工厂方法: 根据模型类型枚举实例化对应的管线子类"""
        target_cls: type[BasePipeline] | None = cls._REGISTRY.get(model_type)
        if target_cls is None:
            raise ValueError(f"未注册的模型管线类型: {model_type}")
        return target_cls()

    def __init__(self) -> None:
        self.pipe: Any = None
        self.strategy: OffloadStrategy = OffloadStrategy.ASYM

    @abstractmethod
    def init_pipeline(
        self,
        model_id: str,
        strategy: OffloadStrategy,
        compile_mode: CompileMode,
    ) -> None:
        """根据卸载策略加载模型权重并初始化内部管线实例"""
        raise NotImplementedError

    @abstractmethod
    def dispatch_pipeline(self, cfg: PipelineConfig) -> None:
        """根据当前配置与卸载策略分发推理任务"""
        raise NotImplementedError

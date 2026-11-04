import os
import sys
import logging
from typing import ClassVar

# ==========================================
# 1. 全局日志配置
# ==========================================
# 独立调试开关: 控制底层调试日志输出 (默认关闭，DEBUG_LOG=1 开启)
ENABLE_DEBUG_LOG: bool = os.environ.get("DEBUG_LOG", "0") == "1"


class ANSIFormatter(logging.Formatter):
    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\033[36m",  # 青色
        logging.INFO: "\033[32m",  # 绿色
        logging.WARNING: "\033[33m",  # 黄色
        logging.ERROR: "\033[31m",  # 红色
    }

    def __init__(self, app_name: str) -> None:
        super().__init__()
        # 将传入的 T2I/Engine 标识强绑定到格式化器中
        self.app_name = app_name

    def format(self, record: logging.LogRecord) -> str:
        color: str = self.COLORS.get(record.levelno, "")
        reset: str = "\033[0m"
        # 视觉隔离: INFO/WARN 保留层级前缀，DEBUG 降级为缩进符号实现像素级对齐
        # 物理拼接，确保终端打印 [T2I][INFO] 而非干瘪的 [INFO]
        level_name: str = (
            f"[{record.levelname}] "
            if record.levelno != logging.DEBUG
            else f"  -->  "
        )
        record.msg = f"{color}{level_name}{record.msg}{reset}"
        return super().format(record)


def setup_pipeline_logger(app_name: str = "") -> logging.Logger:
    # 直接获取 Root Logger ("")
    # 这样下游所有 logging.getLogger(__name__) 冒泡时，必将触发此处的 Handler
    logger: logging.Logger = logging.getLogger()

    # 避免在多次重载时重复挂载 Handler 造成日志重影
    if not logger.handlers:
        # 日志级别控制: 开启时输出 DEBUG 级底层信息，关闭时仅输出 INFO 级业务节点
        logger.setLevel(logging.DEBUG if ENABLE_DEBUG_LOG else logging.INFO)

        # 常规日志输出到 STDOUT
        stdout_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        stdout_handler.setFormatter(ANSIFormatter(app_name))
        logger.addHandler(stdout_handler)

        # 异常日志输出到 STDERR
        stderr_handler: logging.StreamHandler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(ANSIFormatter(app_name))
        logger.addHandler(stderr_handler)

        logger.propagate = False

    return logger

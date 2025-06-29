"""
MODULE: Basic Logger
PURPOSE: 統一ログ記録・監査証跡システム
DEPENDENCIES: structlog, python-json-logger, aals.core.config
INPUT: ログメッセージ、レベル、メタデータ
OUTPUT: 構造化ログファイル、コンソール出力
INTEGRATION: 全モジュールで使用される基盤コンポーネント
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
try:
    from pythonjsonlogger.json import JsonFormatter as jsonlogger_JsonFormatter
except ImportError:
    # 古いバージョン対応
    from pythonjsonlogger import jsonlogger
    jsonlogger_JsonFormatter = jsonlogger.JsonFormatter

from aals.core.config import get_config_manager


class LogLevel(Enum):
    """ログレベル定義"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditAction(Enum):
    """監査アクション定義"""
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS = "access"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    VIEW = "view"
    EXPORT = "export"
    CONFIG_CHANGE = "config_change"
    SECURITY_EVENT = "security_event"


@dataclass
class LogContext:
    """ログコンテキスト情報"""
    module: str
    function: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    environment: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        if self.additional_data:
            data.update(self.additional_data)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class AuditLogEntry:
    """監査ログエントリ"""
    action: AuditAction
    resource: str
    result: str  # success, failure, partial
    details: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    risk_level: str = "low"  # low, medium, high, critical
    compliance_tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        data["action"] = self.action.value
        data["timestamp"] = datetime.now().isoformat()
        return {k: v for k, v in data.items() if v is not None}


class LogFormatter:
    """ログフォーマッター"""
    
    @staticmethod
    def create_json_formatter() -> jsonlogger_JsonFormatter:
        """JSON形式フォーマッターを作成"""
        return jsonlogger_JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    @staticmethod
    def create_text_formatter(colorize: bool = False) -> logging.Formatter:
        """テキスト形式フォーマッターを作成"""
        fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
        
        if colorize:
            # カラー対応フォーマッター（開発時用）
            class ColoredFormatter(logging.Formatter):
                """カラー対応フォーマッター"""
                
                COLORS = {
                    'DEBUG': '\033[36m',     # Cyan
                    'INFO': '\033[32m',      # Green
                    'WARNING': '\033[33m',   # Yellow
                    'ERROR': '\033[31m',     # Red
                    'CRITICAL': '\033[41m',  # Red background
                    'RESET': '\033[0m'       # Reset
                }
                
                def format(self, record):
                    log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
                    record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
                    return super().format(record)
            
            return ColoredFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")


class RotatingFileHandler:
    """ローテーション対応ファイルハンドラー"""
    
    @staticmethod
    def create_handler(
        filepath: str,
        max_size: str = "10MB",
        backup_count: int = 5,
        formatter: Optional[logging.Formatter] = None
    ) -> logging.handlers.RotatingFileHandler:
        """ローテーションファイルハンドラーを作成"""
        
        # ファイルサイズの変換
        size_map = {
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024
        }
        
        max_bytes = 10 * 1024 * 1024  # デフォルト 10MB
        if max_size:
            size_str = max_size.upper()
            for unit, multiplier in size_map.items():
                if size_str.endswith(unit):
                    try:
                        size_value = float(size_str[:-len(unit)])
                        max_bytes = int(size_value * multiplier)
                        break
                    except ValueError:
                        pass
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        if formatter:
            handler.setFormatter(formatter)
        
        return handler


class AALSLogger:
    """AALS統一ログシステム"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.config = self.config_manager.config
        self.module_config = self.config_manager.get_module_config("basic_logger")
        
        if not self.module_config.enabled:
            raise RuntimeError("Basic Logger module is not enabled")
        
        self._setup_structlog()
        self._setup_loggers()
        
        # ログコンテキスト
        self._context = LogContext(module="aals")
        
        self.logger = structlog.get_logger("aals.logger")
        self.logger.info("AALS Logger initialized", module="basic_logger")
    
    def _setup_structlog(self):
        """structlog設定"""
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]
        
        # 開発環境では見やすい形式
        if self.config.is_development:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.processors.JSONRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def _setup_loggers(self):
        """各種ログハンドラーの設定"""
        # ルートロガー設定
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 既存のハンドラーをクリア
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # アプリケーションログハンドラー
        self._setup_app_logger()
        
        # 監査ログハンドラー
        self._setup_audit_logger()
        
        # コンソールハンドラー
        self._setup_console_logger()
    
    def _setup_app_logger(self):
        """アプリケーションログハンドラー設定"""
        app_config = self.module_config.config.get("app_log", {})
        
        filepath = app_config.get("file", "logs/aals.log")
        level = app_config.get("level", "INFO")
        max_size = app_config.get("max_file_size", "10MB")
        backup_count = app_config.get("backup_count", 5)
        format_type = app_config.get("format", "json")
        
        # フォーマッター選択
        if format_type == "json":
            formatter = LogFormatter.create_json_formatter()
        else:
            formatter = LogFormatter.create_text_formatter()
        
        # ハンドラー作成
        handler = RotatingFileHandler.create_handler(
            filepath, max_size, backup_count, formatter
        )
        handler.setLevel(getattr(logging, level.upper()))
        
        # ロガーに追加
        app_logger = logging.getLogger("aals")
        app_logger.addHandler(handler)
        app_logger.setLevel(getattr(logging, level.upper()))
    
    def _setup_audit_logger(self):
        """監査ログハンドラー設定"""
        audit_config = self.module_config.config.get("audit_log", {})
        
        filepath = audit_config.get("file", "logs/audit.log")
        level = audit_config.get("level", "INFO")
        max_size = audit_config.get("max_file_size", "50MB")
        backup_count = audit_config.get("backup_count", 10)
        
        # 監査ログは常にJSON形式
        formatter = LogFormatter.create_json_formatter()
        
        # ハンドラー作成
        handler = RotatingFileHandler.create_handler(
            filepath, max_size, backup_count, formatter
        )
        handler.setLevel(getattr(logging, level.upper()))
        
        # 監査専用ロガー
        audit_logger = logging.getLogger("aals.audit")
        audit_logger.addHandler(handler)
        audit_logger.setLevel(getattr(logging, level.upper()))
        # 親ロガーへの伝播を無効化（重複を防ぐ）
        audit_logger.propagate = False
    
    def _setup_console_logger(self):
        """コンソールハンドラー設定"""
        console_config = self.module_config.config.get("console", {})
        
        if not console_config.get("enabled", True):
            return
        
        level = console_config.get("level", "INFO")
        format_type = console_config.get("format", "text")
        colorize = console_config.get("colorize", False)
        
        # フォーマッター選択
        if format_type == "json":
            formatter = LogFormatter.create_json_formatter()
        else:
            formatter = LogFormatter.create_text_formatter(colorize)
        
        # コンソールハンドラー作成
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(getattr(logging, level.upper()))
        
        # ルートロガーに追加
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
    
    def set_context(self, context: LogContext):
        """ログコンテキストを設定"""
        self._context = context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**context.to_dict())
    
    def update_context(self, **kwargs):
        """ログコンテキストを更新"""
        if self._context.additional_data is None:
            self._context.additional_data = {}
        self._context.additional_data.update(kwargs)
        structlog.contextvars.bind_contextvars(**kwargs)
    
    def get_logger(self, name: str) -> structlog.stdlib.BoundLogger:
        """名前付きロガーを取得"""
        return structlog.get_logger(name)
    
    def debug(self, message: str, **kwargs):
        """DEBUGレベルログ"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """INFOレベルログ"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """WARNINGレベルログ"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """ERRORレベルログ"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """CRITICALレベルログ"""
        self.logger.critical(message, **kwargs)
    
    def audit(self, entry: AuditLogEntry):
        """監査ログ出力"""
        audit_logger = logging.getLogger("aals.audit")
        audit_data = entry.to_dict()
        
        # 構造化ログとして記録
        audit_logger.info("AUDIT", extra=audit_data)
    
    def audit_action(
        self,
        action: AuditAction,
        resource: str,
        result: str = "success",
        details: Optional[str] = None,
        **kwargs
    ):
        """監査アクション記録（簡易版）"""
        entry = AuditLogEntry(
            action=action,
            resource=resource,
            result=result,
            details=details,
            user_id=self._context.user_id,
            session_id=self._context.session_id,
            **kwargs
        )
        self.audit(entry)
    
    def log_exception(self, exc: Exception, context: Optional[str] = None, **kwargs):
        """例外ログ"""
        self.logger.error(
            f"Exception occurred: {exc.__class__.__name__}",
            exception=str(exc),
            context=context,
            exc_info=True,
            **kwargs
        )
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """パフォーマンスログ"""
        self.logger.info(
            f"Performance metric",
            operation=operation,
            duration_seconds=duration,
            **kwargs
        )
    
    def cleanup_old_logs(self, retention_days: Optional[int] = None):
        """古いログファイルのクリーンアップ"""
        retention_days = retention_days or self.module_config.config.get("audit_log", {}).get("retention_days", 90)
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        log_dirs = ["logs"]
        cleaned_files = []
        
        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                continue
                
            for filepath in Path(log_dir).glob("*.log.*"):
                try:
                    if filepath.stat().st_mtime < cutoff_date.timestamp():
                        filepath.unlink()
                        cleaned_files.append(str(filepath))
                except Exception as e:
                    self.warning(f"Failed to cleanup log file {filepath}: {e}")
        
        if cleaned_files:
            self.info(f"Cleaned up {len(cleaned_files)} old log files", files=cleaned_files)


# グローバルロガーインスタンス
_aals_logger: Optional[AALSLogger] = None


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """AALSロガーを取得"""
    global _aals_logger
    if _aals_logger is None:
        _aals_logger = AALSLogger()
    
    if name:
        return _aals_logger.get_logger(name)
    return _aals_logger.logger


def get_aals_logger() -> AALSLogger:
    """AALSロガーインスタンスを取得"""
    global _aals_logger
    if _aals_logger is None:
        _aals_logger = AALSLogger()
    return _aals_logger


def set_log_context(context: LogContext):
    """ログコンテキストを設定"""
    logger = get_aals_logger()
    logger.set_context(context)


def audit_log(entry: AuditLogEntry):
    """監査ログ出力"""
    logger = get_aals_logger()
    logger.audit(entry)
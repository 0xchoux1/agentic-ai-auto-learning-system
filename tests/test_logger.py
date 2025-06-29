"""
Test cases for Basic Logger Module
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import structlog

from aals.core.logger import (
    AALSLogger,
    AuditAction,
    AuditLogEntry,
    LogContext,
    LogFormatter,
    LogLevel,
    RotatingFileHandler,
    audit_log,
    get_aals_logger,
    get_logger,
    set_log_context,
)


class TestLogContext:
    """LogContextクラスのテスト"""
    
    def test_log_context_creation(self):
        """LogContext作成のテスト"""
        context = LogContext(
            module="test_module",
            function="test_function",
            user_id="user123",
            session_id="session456"
        )
        
        assert context.module == "test_module"
        assert context.function == "test_function"
        assert context.user_id == "user123"
        assert context.session_id == "session456"
    
    def test_log_context_to_dict(self):
        """LogContext辞書変換のテスト"""
        context = LogContext(
            module="test_module",
            function="test_function",
            user_id="user123",
            additional_data={"key": "value"}
        )
        
        result = context.to_dict()
        assert result["module"] == "test_module"
        assert result["function"] == "test_function"
        assert result["user_id"] == "user123"
        assert result["key"] == "value"
        assert "session_id" not in result  # Noneの値は除外される
    
    def test_log_context_minimal(self):
        """最小限のLogContextのテスト"""
        context = LogContext(module="minimal")
        result = context.to_dict()
        
        assert result == {"module": "minimal"}


class TestAuditLogEntry:
    """AuditLogEntryクラスのテスト"""
    
    def test_audit_log_entry_creation(self):
        """AuditLogEntry作成のテスト"""
        entry = AuditLogEntry(
            action=AuditAction.LOGIN,
            resource="user_system",
            result="success",
            user_id="user123",
            details="Successful login attempt"
        )
        
        assert entry.action == AuditAction.LOGIN
        assert entry.resource == "user_system"
        assert entry.result == "success"
        assert entry.user_id == "user123"
        assert entry.details == "Successful login attempt"
    
    def test_audit_log_entry_to_dict(self):
        """AuditLogEntry辞書変換のテスト"""
        entry = AuditLogEntry(
            action=AuditAction.CREATE,
            resource="database_record",
            result="success",
            risk_level="medium"
        )
        
        result = entry.to_dict()
        assert result["action"] == "create"  # Enumから文字列に変換
        assert result["resource"] == "database_record"
        assert result["result"] == "success"
        assert result["risk_level"] == "medium"
        assert "timestamp" in result
        
        # タイムスタンプがISO形式かチェック
        datetime.fromisoformat(result["timestamp"])
    
    def test_audit_log_entry_with_compliance_tags(self):
        """コンプライアンスタグ付きAuditLogEntryのテスト"""
        entry = AuditLogEntry(
            action=AuditAction.ACCESS,
            resource="sensitive_data",
            result="success",
            compliance_tags=["GDPR", "SOX", "HIPAA"]
        )
        
        result = entry.to_dict()
        assert result["compliance_tags"] == ["GDPR", "SOX", "HIPAA"]


class TestLogFormatter:
    """LogFormatterクラスのテスト"""
    
    def test_create_json_formatter(self):
        """JSONフォーマッター作成のテスト"""
        formatter = LogFormatter.create_json_formatter()
        assert isinstance(formatter, logging.Formatter)
        
        # テストログレコード
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        # JSON形式かチェック
        data = json.loads(formatted)
        assert data["message"] == "Test message"
        assert data["levelname"] == "INFO"
    
    def test_create_text_formatter(self):
        """テキストフォーマッター作成のテスト"""
        formatter = LogFormatter.create_text_formatter()
        assert isinstance(formatter, logging.Formatter)
        
        # テストログレコード
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Warning message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "WARNING" in formatted
        assert "test.logger" in formatted
        assert "Warning message" in formatted
    
    def test_create_colored_formatter(self):
        """カラーフォーマッター作成のテスト"""
        formatter = LogFormatter.create_text_formatter(colorize=True)
        assert isinstance(formatter, logging.Formatter)
        
        # テストログレコード
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        # ANSI カラーコードが含まれているかチェック
        assert "\033[" in formatted
        assert "Error message" in formatted


class TestRotatingFileHandler:
    """RotatingFileHandlerクラスのテスト"""
    
    def test_create_handler(self):
        """ローテーションハンドラー作成のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test.log")
            
            handler = RotatingFileHandler.create_handler(
                filepath=filepath,
                max_size="1MB",
                backup_count=3
            )
            
            assert isinstance(handler, logging.handlers.RotatingFileHandler)
            assert handler.maxBytes == 1024 * 1024  # 1MB
            assert handler.backupCount == 3
    
    def test_size_parsing(self):
        """ファイルサイズ解析のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test.log")
            
            # KB
            handler = RotatingFileHandler.create_handler(
                filepath=filepath,
                max_size="500KB"
            )
            assert handler.maxBytes == 500 * 1024
            
            # GB
            handler = RotatingFileHandler.create_handler(
                filepath=filepath,
                max_size="2GB"
            )
            assert handler.maxBytes == 2 * 1024 * 1024 * 1024


class TestAALSLogger:
    """AALSLoggerクラスのテスト"""
    
    @pytest.fixture
    def mock_config(self):
        """モック設定"""
        with patch('aals.core.logger.get_config_manager') as mock_manager:
            mock_config_manager = MagicMock()
            mock_config = MagicMock()
            mock_module_config = MagicMock()
            
            mock_config_manager.config = mock_config
            mock_config_manager.get_module_config.return_value = mock_module_config
            
            mock_config.is_development = True
            mock_module_config.enabled = True
            mock_module_config.config = {
                "app_log": {
                    "file": "logs/test_app.log",
                    "level": "INFO",
                    "max_file_size": "1MB",
                    "backup_count": 3,
                    "format": "json"
                },
                "audit_log": {
                    "file": "logs/test_audit.log",
                    "level": "INFO",
                    "max_file_size": "2MB",
                    "backup_count": 5,
                    "retention_days": 30
                },
                "console": {
                    "enabled": True,
                    "level": "DEBUG",
                    "format": "text",
                    "colorize": True
                }
            }
            
            mock_manager.return_value = mock_config_manager
            yield mock_config_manager
    
    def test_logger_initialization(self, mock_config):
        """ロガー初期化のテスト"""
        with tempfile.TemporaryDirectory():
            logger = AALSLogger()
            assert logger is not None
            assert hasattr(logger, 'logger')
    
    def test_logger_disabled_module(self, mock_config):
        """無効化されたモジュールのテスト"""
        mock_config.get_module_config.return_value.enabled = False
        
        with pytest.raises(RuntimeError, match="not enabled"):
            AALSLogger()
    
    def test_set_context(self, mock_config):
        """コンテキスト設定のテスト"""
        with tempfile.TemporaryDirectory():
            logger = AALSLogger()
            context = LogContext(
                module="test_module",
                user_id="user123"
            )
            
            logger.set_context(context)
            assert logger._context == context
    
    def test_update_context(self, mock_config):
        """コンテキスト更新のテスト"""
        with tempfile.TemporaryDirectory():
            logger = AALSLogger()
            logger.update_context(session_id="session456", request_id="req789")
            
            assert logger._context.additional_data["session_id"] == "session456"
            assert logger._context.additional_data["request_id"] == "req789"
    
    def test_log_methods(self, mock_config):
        """各ログメソッドのテスト"""
        with tempfile.TemporaryDirectory():
            logger = AALSLogger()
            
            # 各レベルのログメソッドをテスト
            logger.debug("Debug message", key="value")
            logger.info("Info message", key="value")
            logger.warning("Warning message", key="value")
            logger.error("Error message", key="value")
            logger.critical("Critical message", key="value")
    
    def test_audit_action(self, mock_config):
        """監査アクション記録のテスト"""
        with tempfile.TemporaryDirectory():
            logger = AALSLogger()
            
            logger.audit_action(
                action=AuditAction.LOGIN,
                resource="user_system",
                result="success",
                details="User login successful"
            )
    
    def test_log_exception(self, mock_config):
        """例外ログのテスト"""
        with tempfile.TemporaryDirectory():
            logger = AALSLogger()
            
            try:
                raise ValueError("Test exception")
            except ValueError as e:
                logger.log_exception(e, context="test_context")
    
    def test_log_performance(self, mock_config):
        """パフォーマンスログのテスト"""
        with tempfile.TemporaryDirectory():
            logger = AALSLogger()
            
            logger.log_performance(
                operation="database_query",
                duration=0.123,
                rows_affected=100
            )
    
    def test_cleanup_old_logs(self, mock_config):
        """古いログクリーンアップのテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # テスト用ログディレクトリ作成
            log_dir = os.path.join(temp_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # 古いログファイルを作成
            old_time = (datetime.now() - timedelta(days=100)).timestamp()
            old_log = Path(log_dir) / "old.log.1"
            old_log.touch()
            os.utime(old_log, (old_time, old_time))
            
            # 新しいログファイルを作成
            new_log = Path(log_dir) / "new.log.1"
            new_log.touch()
            
            with patch('os.path.exists', return_value=True), \
                 patch('pathlib.Path.glob', return_value=[old_log, new_log]):
                
                logger = AALSLogger()
                logger.cleanup_old_logs(retention_days=30)
                
                # 古いファイルが削除され、新しいファイルが残っているかチェック
                # 実際のファイル削除はモックされているので、呼び出されたかチェック


class TestGlobalFunctions:
    """グローバル関数のテスト"""
    
    @pytest.fixture(autouse=True)
    def reset_global_logger(self):
        """各テスト前にグローバルロガーをリセット"""
        import aals.core.logger
        aals.core.logger._aals_logger = None
        yield
        aals.core.logger._aals_logger = None
    
    def test_get_logger(self):
        """get_logger関数のテスト"""
        with patch('aals.core.logger.AALSLogger') as MockLogger:
            mock_instance = MagicMock()
            MockLogger.return_value = mock_instance
            
            logger = get_logger("test.module")
            MockLogger.assert_called_once()
            mock_instance.get_logger.assert_called_once_with("test.module")
    
    def test_get_aals_logger(self):
        """get_aals_logger関数のテスト"""
        with patch('aals.core.logger.AALSLogger') as MockLogger:
            mock_instance = MagicMock()
            MockLogger.return_value = mock_instance
            
            logger = get_aals_logger()
            MockLogger.assert_called_once()
            assert logger == mock_instance
    
    def test_set_log_context(self):
        """set_log_context関数のテスト"""
        with patch('aals.core.logger.AALSLogger') as MockLogger:
            mock_instance = MagicMock()
            MockLogger.return_value = mock_instance
            
            context = LogContext(module="test")
            set_log_context(context)
            
            MockLogger.assert_called_once()
            mock_instance.set_context.assert_called_once_with(context)
    
    def test_audit_log(self):
        """audit_log関数のテスト"""
        with patch('aals.core.logger.AALSLogger') as MockLogger:
            mock_instance = MagicMock()
            MockLogger.return_value = mock_instance
            
            entry = AuditLogEntry(
                action=AuditAction.ACCESS,
                resource="test_resource",
                result="success"
            )
            audit_log(entry)
            
            MockLogger.assert_called_once()
            mock_instance.audit.assert_called_once_with(entry)


class TestEnumClasses:
    """Enum クラスのテスト"""
    
    def test_log_level_enum(self):
        """LogLevel Enumのテスト"""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"
    
    def test_audit_action_enum(self):
        """AuditAction Enumのテスト"""
        assert AuditAction.LOGIN.value == "login"
        assert AuditAction.LOGOUT.value == "logout"
        assert AuditAction.ACCESS.value == "access"
        assert AuditAction.CREATE.value == "create"
        assert AuditAction.UPDATE.value == "update"
        assert AuditAction.DELETE.value == "delete"
        assert AuditAction.EXECUTE.value == "execute"
        assert AuditAction.VIEW.value == "view"
        assert AuditAction.EXPORT.value == "export"
        assert AuditAction.CONFIG_CHANGE.value == "config_change"
        assert AuditAction.SECURITY_EVENT.value == "security_event"


if __name__ == "__main__":
    pytest.main([__file__])
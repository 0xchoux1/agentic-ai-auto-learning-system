"""
Test cases for Config Manager
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from aals.core.config import (
    AALSConfig,
    ConfigManager,
    DatabaseConfig,
    IntegrationConfig,
    ModuleConfig,
    RedisConfig,
    SecurityConfig,
    get_config,
    get_config_manager,
)


class TestDatabaseConfig:
    """データベース設定のテスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "aals"
        assert config.username == "aals_user"
        assert config.password == ""
    
    def test_connection_string(self):
        """接続文字列生成のテスト"""
        config = DatabaseConfig(
            host="testhost",
            port=5433,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        expected = "postgresql://testuser:testpass@testhost:5433/testdb"
        assert config.connection_string == expected
    
    def test_async_connection_string(self):
        """非同期接続文字列生成のテスト"""
        config = DatabaseConfig(
            host="testhost",
            port=5433,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        expected = "postgresql+asyncpg://testuser:testpass@testhost:5433/testdb"
        assert config.async_connection_string == expected


class TestRedisConfig:
    """Redis設定のテスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.database == 0
        assert config.password == ""
    
    def test_connection_string_without_password(self):
        """パスワードなし接続文字列のテスト"""
        config = RedisConfig(host="testhost", port=6380, database=1)
        expected = "redis://testhost:6380/1"
        assert config.connection_string == expected
    
    def test_connection_string_with_password(self):
        """パスワードあり接続文字列のテスト"""
        config = RedisConfig(
            host="testhost",
            port=6380,
            database=1,
            password="testpass"
        )
        expected = "redis://:testpass@testhost:6380/1"
        assert config.connection_string == expected


class TestSecurityConfig:
    """セキュリティ設定のテスト"""
    
    def test_valid_secret_key(self):
        """有効なシークレットキーのテスト"""
        secret_key = "a" * 32  # 32文字のキー
        config = SecurityConfig(secret_key=secret_key)
        assert config.secret_key == secret_key
    
    def test_short_secret_key(self):
        """短いシークレットキーのテスト"""
        with pytest.raises(ValueError, match="at least 32 characters"):
            SecurityConfig(secret_key="short")
    
    def test_empty_secret_key(self):
        """空のシークレットキーのテスト"""
        with pytest.raises(ValueError, match="Secret key is required"):
            SecurityConfig(secret_key="")


class TestAALSConfig:
    """AALS設定のテスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        with patch.dict(os.environ, {"AALS_SECRET_KEY": "a" * 32}, clear=True):
            config = AALSConfig()
            assert config.environment == "development"
            assert config.debug is True
            assert config.log_level == "INFO"
            assert config.version == "0.1.0"
    
    def test_environment_properties(self):
        """環境判定プロパティのテスト"""
        with patch.dict(os.environ, {"AALS_SECRET_KEY": "a" * 32}, clear=True):
            # Development
            config = AALSConfig(environment="development")
            assert config.is_development is True
            assert config.is_production is False
            assert config.is_staging is False
            
            # Production
            config = AALSConfig(environment="production")
            assert config.is_development is False
            assert config.is_production is True
            assert config.is_staging is False
            
            # Staging
            config = AALSConfig(environment="staging")
            assert config.is_development is False
            assert config.is_production is False
            assert config.is_staging is True
    
    def test_database_property(self):
        """データベースプロパティのテスト"""
        with patch.dict(os.environ, {
            "AALS_SECRET_KEY": "a" * 32,
            "AALS_DB_HOST": "testhost",
            "AALS_DB_PORT": "5433",
            "AALS_DB_PASSWORD": "testpass"
        }, clear=True):
            config = AALSConfig()
            db_config = config.database
            assert isinstance(db_config, DatabaseConfig)
            assert db_config.host == "testhost"
            assert db_config.port == 5433
            assert db_config.password == "testpass"
    
    def test_redis_property(self):
        """Redisプロパティのテスト"""
        with patch.dict(os.environ, {
            "AALS_SECRET_KEY": "a" * 32,
            "AALS_REDIS_HOST": "testhost",
            "AALS_REDIS_PORT": "6380",
            "AALS_REDIS_PASSWORD": "testpass"
        }, clear=True):
            config = AALSConfig()
            redis_config = config.redis
            assert isinstance(redis_config, RedisConfig)
            assert redis_config.host == "testhost"
            assert redis_config.port == 6380
            assert redis_config.password == "testpass"


class TestConfigManager:
    """ConfigManagerのテスト"""
    
    def create_test_config_file(self, config_data: dict) -> str:
        """テスト用設定ファイルを作成"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_data, temp_file, default_flow_style=False)
        temp_file.close()
        return temp_file.name
    
    def test_load_yaml_config(self):
        """YAML設定ファイル読み込みのテスト"""
        config_data = {
            "system": {"name": "test_aals"},
            "modules": {
                "test_module": {
                    "enabled": True,
                    "setting1": "value1"
                }
            }
        }
        
        config_file = self.create_test_config_file(config_data)
        
        try:
            with patch.dict(os.environ, {"AALS_SECRET_KEY": "a" * 32}, clear=True):
                manager = ConfigManager(config_file)
                module_config = manager.get_module_config("test_module")
                assert module_config.enabled is True
                assert module_config.config["setting1"] == "value1"
        finally:
            os.unlink(config_file)
    
    def test_module_enabled_check(self):
        """モジュール有効化チェックのテスト"""
        config_data = {
            "modules": {
                "enabled_module": {"enabled": True},
                "disabled_module": {"enabled": False}
            }
        }
        
        config_file = self.create_test_config_file(config_data)
        
        try:
            with patch.dict(os.environ, {"AALS_SECRET_KEY": "a" * 32}, clear=True):
                manager = ConfigManager(config_file)
                assert manager.is_module_enabled("enabled_module") is True
                assert manager.is_module_enabled("disabled_module") is False
                assert manager.is_module_enabled("nonexistent_module") is False
        finally:
            os.unlink(config_file)
    
    def test_integration_config(self):
        """統合設定のテスト"""
        config_data = {
            "integrations": {
                "slack": {
                    "token": "test_token",
                    "channels": ["#alerts"]
                }
            }
        }
        
        config_file = self.create_test_config_file(config_data)
        
        try:
            with patch.dict(os.environ, {"AALS_SECRET_KEY": "a" * 32}, clear=True):
                manager = ConfigManager(config_file)
                slack_config = manager.get_integration_config("slack")
                assert slack_config["token"] == "test_token"
                assert slack_config["channels"] == ["#alerts"]
        finally:
            os.unlink(config_file)
    
    def test_config_validation(self):
        """設定検証のテスト"""
        config_file = self.create_test_config_file({})
        
        try:
            # 本番環境でデバッグモード有効の場合をテスト
            with patch.dict(os.environ, {
                "AALS_SECRET_KEY": "a" * 32,
                "AALS_ENVIRONMENT": "production",
                "AALS_DEBUG": "true"
            }):
                manager = ConfigManager(config_file)
                errors = manager.validate_config()
                assert any("Debug mode should be disabled" in error for error in errors)
                
            # 本番環境でDEBUGログレベルの場合をテスト
            with patch.dict(os.environ, {
                "AALS_SECRET_KEY": "a" * 32,
                "AALS_ENVIRONMENT": "production",
                "AALS_DEBUG": "false",
                "AALS_LOG_LEVEL": "DEBUG"
            }):
                manager = ConfigManager(config_file)
                errors = manager.validate_config()
                assert any("Log level should not be DEBUG" in error for error in errors)
        finally:
            os.unlink(config_file)
    
    def test_nonexistent_config_file(self):
        """存在しない設定ファイルのテスト"""
        with patch.dict(os.environ, {"AALS_SECRET_KEY": "a" * 32}, clear=True):
            manager = ConfigManager("nonexistent.yaml")
            # エラーにならずにデフォルト設定が使用される
            assert manager.config is not None


class TestGlobalFunctions:
    """グローバル関数のテスト"""
    
    def test_get_config_manager_singleton(self):
        """ConfigManagerシングルトンのテスト"""
        # グローバル変数をリセット
        import aals.core.config
        aals.core.config._config_manager = None
        
        with patch.dict(os.environ, {"AALS_SECRET_KEY": "a" * 32}, clear=True):
            manager1 = get_config_manager()
            manager2 = get_config_manager()
            assert manager1 is manager2
    
    def test_get_config(self):
        """get_config関数のテスト"""
        with patch.dict(os.environ, {"AALS_SECRET_KEY": "a" * 32}, clear=True):
            config = get_config()
            assert isinstance(config, AALSConfig)


if __name__ == "__main__":
    pytest.main([__file__])
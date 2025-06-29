"""
MODULE: Config Manager
PURPOSE: 設定ファイル管理、環境変数統合、バリデーション機能
DEPENDENCIES: pydantic, pydantic-settings, python-dotenv, pyyaml
INPUT: YAML設定ファイル, 環境変数
OUTPUT: 型安全な設定オブジェクト
INTEGRATION: 全モジュールで使用される基盤コンポーネント
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    """データベース設定"""
    host: str = "localhost"
    port: int = 5432
    database: str = "aals"
    username: str = "aals_user"
    password: str = ""
    
    @property
    def connection_string(self) -> str:
        """PostgreSQL接続文字列を生成"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        """非同期PostgreSQL接続文字列を生成"""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    """Redis設定"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: str = ""
    
    @property
    def connection_string(self) -> str:
        """Redis接続文字列を生成"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"redis://{self.host}:{self.port}/{self.database}"


class ModuleConfig(BaseModel):
    """モジュール設定"""
    enabled: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)


class SecurityConfig(BaseModel):
    """セキュリティ設定"""
    secret_key: str = ""
    rate_limit_enabled: bool = True
    requests_per_minute: int = 60
    audit_enabled: bool = True
    audit_log_file: str = "logs/audit.log"
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v):
        if not v:
            raise ValueError("Secret key is required")
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class IntegrationConfig(BaseModel):
    """外部サービス統合設定"""
    slack_token: str = ""
    github_token: str = ""
    claude_api_key: str = ""
    openai_api_key: str = ""
    ssh_key_file: str = ""
    ssh_known_hosts_file: str = ""


class AALSConfig(BaseSettings):
    """AALS システム設定"""
    
    model_config = SettingsConfigDict(
        env_prefix="AALS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # システム基本設定
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    version: str = "0.1.0"
    
    # データベース設定
    db_host: str = "localhost"
    db_port: int = 5432
    db_database: str = "aals"
    db_username: str = "aals_user"
    db_password: str = ""
    
    # Redis設定
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_database: int = 0
    redis_password: str = ""
    
    # セキュリティ設定
    secret_key: str = ""
    
    # 統合設定
    slack_token: str = ""
    github_token: str = ""
    claude_api_key: str = ""
    openai_api_key: str = ""
    ssh_key_file: str = ""
    ssh_known_hosts_file: str = ""
    
    # 設定ファイルパス
    config_file: str = "config/default.yaml"
    
    @property
    def database(self) -> DatabaseConfig:
        """データベース設定を取得"""
        return DatabaseConfig(
            host=self.db_host,
            port=self.db_port,
            database=self.db_database,
            username=self.db_username,
            password=self.db_password
        )
    
    @property
    def redis(self) -> RedisConfig:
        """Redis設定を取得"""
        return RedisConfig(
            host=self.redis_host,
            port=self.redis_port,
            database=self.redis_database,
            password=self.redis_password
        )
    
    @property
    def security(self) -> SecurityConfig:
        """セキュリティ設定を取得"""
        try:
            return SecurityConfig(secret_key=self.secret_key)
        except ValueError:
            # バリデーションエラーは validate_config で処理
            return SecurityConfig(secret_key="invalid_key_placeholder_32_chars")
    
    @property
    def integrations(self) -> IntegrationConfig:
        """統合設定を取得"""
        return IntegrationConfig(
            slack_token=self.slack_token,
            github_token=self.github_token,
            claude_api_key=self.claude_api_key,
            openai_api_key=self.openai_api_key,
            ssh_key_file=self.ssh_key_file,
            ssh_known_hosts_file=self.ssh_known_hosts_file
        )
    
    @property
    def is_production(self) -> bool:
        """本番環境かどうか"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """開発環境かどうか"""
        return self.environment.lower() == "development"
    
    @property
    def is_staging(self) -> bool:
        """ステージング環境かどうか"""
        return self.environment.lower() == "staging"


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, config_file: Optional[str] = None):
        self._config_file = config_file or "config/default.yaml"
        self._config: Optional[AALSConfig] = None
        self._yaml_config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """設定ファイルを読み込み"""
        # YAML設定ファイルを読み込み
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    self._yaml_config = yaml.safe_load(f) or {}
            except Exception as e:
                raise RuntimeError(f"Failed to load config file {self._config_file}: {e}")
        
        # 環境変数から設定を読み込み（YAML設定を上書き）
        try:
            self._config = AALSConfig(_env_file=".env")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    @property
    def config(self) -> AALSConfig:
        """メイン設定を取得"""
        if self._config is None:
            self._load_config()
        return self._config
    
    def get_module_config(self, module_name: str) -> ModuleConfig:
        """モジュール設定を取得"""
        module_config = self._yaml_config.get("modules", {}).get(module_name, {})
        return ModuleConfig(
            enabled=module_config.get("enabled", False),
            config=module_config
        )
    
    def is_module_enabled(self, module_name: str) -> bool:
        """モジュールが有効かどうか"""
        return self.get_module_config(module_name).enabled
    
    def get_integration_config(self, service_name: str) -> Dict[str, Any]:
        """統合サービス設定を取得"""
        return self._yaml_config.get("integrations", {}).get(service_name, {})
    
    def validate_config(self) -> List[str]:
        """設定の妥当性をチェック"""
        errors = []
        
        # 必須設定のチェック
        if not self.config.secret_key:
            errors.append("Secret key is required (set AALS_SECRET_KEY)")
        
        # 本番環境でのセキュリティチェック
        if self.config.is_production:
            if self.config.debug:
                errors.append("Debug mode should be disabled in production")
            if self.config.log_level == "DEBUG":
                errors.append("Log level should not be DEBUG in production")
        
        # データベース接続チェック
        if not self.config.db_password and not self.config.is_development:
            errors.append("Database password is required for non-development environments")
        
        return errors
    
    def reload(self) -> None:
        """設定を再読み込み"""
        self._config = None
        self._yaml_config = {}
        self._load_config()
    
    def get_log_config(self) -> Dict[str, Any]:
        """ログ設定を取得"""
        return {
            "level": self.config.log_level,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": self._yaml_config.get("modules", {}).get("basic_logger", {}).get("log_file", "logs/aals.log")
        }


# グローバル設定インスタンス
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """設定管理インスタンスを取得（シングルトン）"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager


def get_config() -> AALSConfig:
    """設定を取得"""
    return get_config_manager().config
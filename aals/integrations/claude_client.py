"""
Claude API Client

このモジュールはClaude APIとの統合機能を提供します。
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
import asyncio
from aiohttp import ClientTimeout, ClientResponseError

from aals.core.logger import get_logger

logger = get_logger(__name__)


class ClaudeModel(Enum):
    """利用可能なClaudeモデル"""
    HAIKU = "claude-3-haiku-20240307"
    SONNET = "claude-3-sonnet-20240229"
    OPUS = "claude-3-opus-20240229"


@dataclass
class ClaudeMessage:
    """Claudeメッセージ"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ClaudeResponse:
    """Claude API レスポンス"""
    content: str
    model: str
    usage: Dict[str, int]
    stop_reason: Optional[str] = None
    request_id: Optional[str] = None
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ClaudeAPIConfig:
    """Claude API設定"""
    api_key: str
    model: str = ClaudeModel.HAIKU.value
    max_tokens: int = 4096
    temperature: float = 0.1
    base_url: str = "https://api.anthropic.com"
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


class ClaudeAPIError(Exception):
    """Claude API エラー"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class ClaudeAPIClient:
    """Claude API クライアント"""
    
    def __init__(self, config: ClaudeAPIConfig):
        """初期化"""
        self.config = config
        self.session = None
        self._last_request_time = 0.0
        self._request_count = 0
        
        # APIキーの検証
        if not self.config.api_key:
            raise ValueError("Claude API key is required")
            
        logger.info("Claude API Client initialized", 
                   model=self.config.model,
                   max_tokens=self.config.max_tokens,
                   has_api_key=bool(self.config.api_key))
    
    async def __aenter__(self):
        """非同期コンテキストマネージャー開始"""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャー終了"""
        await self._close_session()
    
    async def _create_session(self):
        """HTTPセッション作成"""
        if self.session is None:
            timeout = ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _close_session(self):
        """HTTPセッション終了"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _get_headers(self) -> Dict[str, str]:
        """リクエストヘッダー取得"""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def _format_messages(self, messages: List[ClaudeMessage]) -> List[Dict[str, str]]:
        """メッセージ形式の変換"""
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    
    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """API リクエスト実行"""
        if not self.session:
            await self._create_session()
        
        url = f"{self.config.base_url}/v1/messages"
        headers = self._get_headers()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                
                # レート制限の考慮（簡単な実装）
                now = time.time()
                if now - self._last_request_time < 1.0:  # 1秒間隔
                    await asyncio.sleep(1.0 - (now - self._last_request_time))
                
                async with self.session.post(url, headers=headers, json=payload) as response:
                    response_time = time.time() - start_time
                    self._last_request_time = time.time()
                    self._request_count += 1
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Claude API request successful",
                                   response_time=response_time,
                                   request_count=self._request_count)
                        return result
                    else:
                        error_text = await response.text()
                        logger.error("Claude API request failed",
                                   status_code=response.status,
                                   error=error_text,
                                   attempt=attempt + 1)
                        
                        if attempt < self.config.max_retries:
                            await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                            continue
                        else:
                            raise ClaudeAPIError(
                                f"HTTP {response.status}: {error_text}",
                                status_code=response.status,
                                response=error_text
                            )
                            
            except aiohttp.ClientError as e:
                logger.error("Claude API network error",
                           error=str(e),
                           attempt=attempt + 1)
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise ClaudeAPIError(f"Network error: {str(e)}")
    
    async def send_message(
        self,
        messages: List[ClaudeMessage],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> ClaudeResponse:
        """メッセージ送信"""
        start_time = time.time()
        
        # パラメータの設定
        model = model or self.config.model
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        # リクエストペイロード構築
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._format_messages(messages)
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        logger.info("Sending Claude API request",
                   model=model,
                   message_count=len(messages),
                   has_system_prompt=bool(system_prompt))
        
        try:
            # API リクエスト実行
            result = await self._make_request(payload)
            
            # レスポンス解析
            content = result.get("content", [])
            if content and isinstance(content, list) and len(content) > 0:
                response_text = content[0].get("text", "")
            else:
                response_text = ""
            
            response = ClaudeResponse(
                content=response_text,
                model=result.get("model", model),
                usage=result.get("usage", {}),
                stop_reason=result.get("stop_reason"),
                request_id=result.get("id"),
                response_time=time.time() - start_time
            )
            
            logger.info("Claude API response received",
                       content_length=len(response.content),
                       usage=response.usage,
                       response_time=response.response_time)
            
            return response
            
        except Exception as e:
            logger.error("Claude API request failed",
                        error=str(e),
                        exception_type=type(e).__name__)
            raise
    
    async def send_single_message(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ClaudeResponse:
        """単一メッセージ送信（便利メソッド）"""
        messages = [ClaudeMessage(role="user", content=user_message)]
        return await self.send_message(messages, system_prompt=system_prompt, **kwargs)
    
    async def verify_connection(self) -> bool:
        """接続確認"""
        try:
            test_message = "Hello, can you respond with just 'OK'?"
            response = await self.send_single_message(test_message, max_tokens=10)
            
            logger.info("Claude API connection verified",
                       response_content=response.content[:50])
            return True
            
        except Exception as e:
            logger.error("Claude API connection verification failed",
                        error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            "request_count": self._request_count,
            "last_request_time": self._last_request_time,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
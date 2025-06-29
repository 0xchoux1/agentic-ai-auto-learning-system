"""
Slack API Client
Slack APIとの統合を担当するクライアント
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from aals.core.config import get_config


@dataclass
class SlackMessage:
    """Slackメッセージのデータクラス"""
    channel: str
    channel_name: str
    timestamp: str
    user: Optional[str]
    text: str
    thread_ts: Optional[str] = None
    reactions: List[Dict[str, Any]] = None
    is_alert: bool = False
    alert_level: Optional[str] = None
    
    def __post_init__(self):
        if self.reactions is None:
            self.reactions = []
    
    @property
    def datetime(self) -> datetime:
        """タイムスタンプをdatetimeオブジェクトに変換"""
        return datetime.fromtimestamp(float(self.timestamp))
    
    @property
    def message_url(self) -> str:
        """メッセージのURLを生成"""
        ts_clean = self.timestamp.replace('.', '')
        return f"https://slack.com/app_redirect?channel={self.channel}&message_ts={ts_clean}"


class SlackAPIClient:
    """Slack API クライアント"""
    
    def __init__(self, token: Optional[str] = None):
        self.config = get_config()
        self.token = token or self.config.slack_token
        
        if not self.token:
            raise ValueError("Slack token is required. Set AALS_SLACK_TOKEN environment variable.")
        
        self.client = WebClient(token=self.token)
        self.logger = logging.getLogger(__name__)
        
        # キャッシュ管理
        self._channel_cache: Dict[str, str] = {}  # channel_name -> channel_id
        self._cache_timestamp: Optional[datetime] = None
    
    async def verify_connection(self) -> bool:
        """Slack API接続確認"""
        try:
            response = self.client.auth_test()
            if response["ok"]:
                self.logger.info(f"Slack connection verified. Bot user: {response.get('user', 'Unknown')}")
                return True
            else:
                self.logger.error(f"Slack auth failed: {response.get('error', 'Unknown error')}")
                return False
        except SlackApiError as e:
            self.logger.error(f"Slack API error: {e.response['error']}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during Slack auth: {e}")
            return False
    
    async def get_channel_id(self, channel_name: str) -> Optional[str]:
        """チャンネル名からチャンネルIDを取得"""
        # チャンネル名の正規化
        if channel_name.startswith('#'):
            channel_name = channel_name[1:]
        
        # キャッシュから取得
        if channel_name in self._channel_cache:
            return self._channel_cache[channel_name]
        
        try:
            # パブリックチャンネルを取得
            response = self.client.conversations_list(
                types="public_channel,private_channel",
                limit=1000
            )
            
            if response["ok"]:
                for channel in response["channels"]:
                    name = channel["name"]
                    channel_id = channel["id"]
                    self._channel_cache[name] = channel_id
                    
                return self._channel_cache.get(channel_name)
            else:
                self.logger.error(f"Failed to get channels: {response.get('error')}")
                return None
                
        except SlackApiError as e:
            self.logger.error(f"Slack API error getting channel ID: {e.response['error']}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting channel ID: {e}")
            return None
    
    async def get_channel_messages(
        self, 
        channel_name: str, 
        limit: int = 100,
        hours_back: int = 24
    ) -> List[SlackMessage]:
        """チャンネルからメッセージを取得"""
        channel_id = await self.get_channel_id(channel_name)
        if not channel_id:
            self.logger.warning(f"Channel not found: {channel_name}")
            return []
        
        # 時間範囲を計算
        oldest = datetime.now() - timedelta(hours=hours_back)
        oldest_timestamp = oldest.timestamp()
        
        try:
            response = self.client.conversations_history(
                channel=channel_id,
                limit=limit,
                oldest=str(oldest_timestamp),
                inclusive=True
            )
            
            if not response["ok"]:
                self.logger.error(f"Failed to get messages from {channel_name}: {response.get('error')}")
                return []
            
            messages = []
            for msg in response["messages"]:
                # ボットメッセージやシステムメッセージをフィルタ
                if msg.get("subtype") in ["bot_message", "channel_join", "channel_leave"]:
                    continue
                
                slack_msg = SlackMessage(
                    channel=channel_id,
                    channel_name=channel_name,
                    timestamp=msg["ts"],
                    user=msg.get("user"),
                    text=msg.get("text", ""),
                    thread_ts=msg.get("thread_ts"),
                    reactions=msg.get("reactions", [])
                )
                
                # アラートメッセージかどうかを判定
                slack_msg.is_alert, slack_msg.alert_level = self._analyze_alert_message(slack_msg.text)
                
                messages.append(slack_msg)
            
            self.logger.info(f"Retrieved {len(messages)} messages from {channel_name}")
            return messages
            
        except SlackApiError as e:
            self.logger.error(f"Slack API error getting messages: {e.response['error']}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error getting messages: {e}")
            return []
    
    def _analyze_alert_message(self, text: str) -> tuple[bool, Optional[str]]:
        """メッセージがアラートかどうかを分析"""
        if not text:
            return False, None
        
        text_upper = text.upper()
        
        # 特定のキーワードでの詳細チェック（優先順位順）
        # より具体的なキーワードを先にチェック
        specific_keywords = [
            ("CRITICAL", "critical"),
            ("WARNING", "warning"),
            ("WARN", "warning"),
            ("ERROR", "error"),
            ("FAILED", "error"),
            ("DOWN", "critical"),
            ("HIGH", "high")
        ]
        
        for keyword, level in specific_keywords:
            if keyword in text_upper:
                return True, level
        
        # 一般的なパターンマッチング（infoレベル）
        alert_patterns = [
            "FIRING",
            "RESOLVED",
            "PROMETHEUS",
            "MONITORING",
            "NAGIOS",
            "GRAFANA",
            "CPU USAGE",
            "MEMORY USAGE",
            "DISK SPACE",
            "NETWORK",
            "SERVICE DOWN",
            "TIMEOUT",
            "ALERT"  # 最後にチェック
        ]
        
        for pattern in alert_patterns:
            if pattern in text_upper:
                return True, "info"
        
        return False, None
    
    async def get_all_alert_messages(
        self, 
        channels: List[str], 
        hours_back: int = 24,
        max_messages_per_channel: int = 100
    ) -> List[SlackMessage]:
        """複数チャンネルからアラートメッセージを取得"""
        all_messages = []
        
        tasks = []
        for channel in channels:
            task = self.get_channel_messages(
                channel, 
                limit=max_messages_per_channel,
                hours_back=hours_back
            )
            tasks.append(task)
        
        # 並行でメッセージを取得
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error getting messages from {channels[i]}: {result}")
                continue
            
            # アラートメッセージのみフィルタ
            alert_messages = [msg for msg in result if msg.is_alert]
            all_messages.extend(alert_messages)
        
        # タイムスタンプで並び替え（新しいものが先）
        all_messages.sort(key=lambda x: float(x.timestamp), reverse=True)
        
        self.logger.info(f"Total alert messages found: {len(all_messages)}")
        return all_messages
    
    async def get_thread_replies(self, channel_id: str, thread_ts: str) -> List[SlackMessage]:
        """スレッドの返信を取得"""
        try:
            response = self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts
            )
            
            if not response["ok"]:
                self.logger.error(f"Failed to get thread replies: {response.get('error')}")
                return []
            
            replies = []
            for msg in response["messages"][1:]:  # 最初のメッセージは除く
                slack_msg = SlackMessage(
                    channel=channel_id,
                    channel_name="",  # チャンネル名は別途取得が必要
                    timestamp=msg["ts"],
                    user=msg.get("user"),
                    text=msg.get("text", ""),
                    thread_ts=thread_ts,
                    reactions=msg.get("reactions", [])
                )
                replies.append(slack_msg)
            
            return replies
            
        except SlackApiError as e:
            self.logger.error(f"Slack API error getting thread replies: {e.response['error']}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error getting thread replies: {e}")
            return []
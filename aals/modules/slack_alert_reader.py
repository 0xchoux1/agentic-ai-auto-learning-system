"""
MODULE: Slack Alert Reader
PURPOSE: Slackからアラート情報を取得・分析
DEPENDENCIES: slack-sdk, aals.core.config, aals.integrations.slack_client
INPUT: Slackチャンネル, 時間範囲
OUTPUT: 構造化されたアラートデータ（JSON）
INTEGRATION: Module 7 (Alert Correlator) で使用
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from aals.core.config import get_config_manager
from aals.integrations.slack_client import SlackAPIClient, SlackMessage


@dataclass
class AlertSummary:
    """アラート要約データ"""
    total_alerts: int
    critical_count: int
    warning_count: int
    error_count: int
    info_count: int
    channels: List[str]
    time_range: Dict[str, str]
    most_active_channel: str
    latest_alert: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass  
class ChannelAnalysis:
    """チャンネル分析結果"""
    channel_name: str
    message_count: int
    alert_count: int
    alert_rate: float
    most_common_alert_level: str
    first_alert_time: Optional[str]
    last_alert_time: Optional[str]
    keywords_frequency: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


class SlackAlertReader:
    """Slack Alert Reader モジュール"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.config = self.config_manager.config
        self.module_config = self.config_manager.get_module_config("slack_alert_reader")
        
        if not self.module_config.enabled:
            raise RuntimeError("Slack Alert Reader module is not enabled")
        
        self.slack_client = SlackAPIClient()
        self.logger = logging.getLogger(__name__)
        
        # モジュール設定を取得
        self.channels = self.module_config.config.get("channels", [])
        self.max_messages = self.module_config.config.get("max_messages", 100)
        self.lookback_hours = self.module_config.config.get("lookback_hours", 24)
        self.alert_keywords = self.module_config.config.get("alert_keywords", [])
        
        self.logger.info(f"Slack Alert Reader initialized. Monitoring channels: {self.channels}")
    
    async def verify_setup(self) -> bool:
        """モジュールのセットアップを確認"""
        try:
            # Slack接続確認
            if not await self.slack_client.verify_connection():
                self.logger.error("Slack connection verification failed")
                return False
            
            # チャンネル存在確認
            missing_channels = []
            for channel in self.channels:
                channel_id = await self.slack_client.get_channel_id(channel)
                if not channel_id:
                    missing_channels.append(channel)
            
            if missing_channels:
                self.logger.warning(f"Some channels not found: {missing_channels}")
                return False
            
            self.logger.info("Slack Alert Reader setup verification completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Setup verification failed: {e}")
            return False
    
    async def get_recent_alerts(
        self, 
        hours_back: Optional[int] = None,
        channels: Optional[List[str]] = None
    ) -> List[SlackMessage]:
        """最近のアラートを取得"""
        hours_back = hours_back or self.lookback_hours
        channels = channels or self.channels
        
        try:
            self.logger.info(f"Fetching alerts from {len(channels)} channels, {hours_back} hours back")
            
            messages = await self.slack_client.get_all_alert_messages(
                channels=channels,
                hours_back=hours_back,
                max_messages_per_channel=self.max_messages
            )
            
            self.logger.info(f"Retrieved {len(messages)} alert messages")
            return messages
            
        except Exception as e:
            self.logger.error(f"Error fetching recent alerts: {e}")
            return []
    
    def analyze_alert_patterns(self, messages: List[SlackMessage]) -> AlertSummary:
        """アラートパターンを分析"""
        if not messages:
            return AlertSummary(
                total_alerts=0,
                critical_count=0,
                warning_count=0,
                error_count=0,
                info_count=0,
                channels=[],
                time_range={},
                most_active_channel=""
            )
        
        # アラートレベル別カウント
        level_counts = {"critical": 0, "warning": 0, "error": 0, "info": 0}
        channel_counts = {}
        
        for msg in messages:
            if msg.alert_level:
                level_counts[msg.alert_level] = level_counts.get(msg.alert_level, 0) + 1
            
            channel_counts[msg.channel_name] = channel_counts.get(msg.channel_name, 0) + 1
        
        # 最もアクティブなチャンネル
        most_active_channel = max(channel_counts, key=channel_counts.get) if channel_counts else ""
        
        # 時間範囲
        timestamps = [float(msg.timestamp) for msg in messages]
        time_range = {
            "start": datetime.fromtimestamp(min(timestamps)).isoformat() if timestamps else "",
            "end": datetime.fromtimestamp(max(timestamps)).isoformat() if timestamps else ""
        }
        
        # 最新のアラート
        latest_alert = None
        if messages:
            latest_msg = messages[0]  # 既にタイムスタンプでソート済み
            latest_alert = {
                "channel": latest_msg.channel_name,
                "text": latest_msg.text,
                "timestamp": latest_msg.timestamp,
                "level": latest_msg.alert_level,
                "url": latest_msg.message_url
            }
        
        return AlertSummary(
            total_alerts=len(messages),
            critical_count=level_counts["critical"],
            warning_count=level_counts["warning"],
            error_count=level_counts["error"],
            info_count=level_counts["info"],
            channels=list(channel_counts.keys()),
            time_range=time_range,
            most_active_channel=most_active_channel,
            latest_alert=latest_alert
        )
    
    def analyze_channel(self, channel_name: str, messages: List[SlackMessage]) -> ChannelAnalysis:
        """特定チャンネルの分析"""
        channel_messages = [msg for msg in messages if msg.channel_name == channel_name]
        alert_messages = [msg for msg in channel_messages if msg.is_alert]
        
        if not channel_messages:
            return ChannelAnalysis(
                channel_name=channel_name,
                message_count=0,
                alert_count=0,
                alert_rate=0.0,
                most_common_alert_level="",
                first_alert_time=None,
                last_alert_time=None,
                keywords_frequency={}
            )
        
        # アラート率計算
        alert_rate = (len(alert_messages) / len(channel_messages)) * 100
        
        # 最も一般的なアラートレベル
        level_counts = {}
        for msg in alert_messages:
            if msg.alert_level:
                level_counts[msg.alert_level] = level_counts.get(msg.alert_level, 0) + 1
        
        most_common_level = max(level_counts, key=level_counts.get) if level_counts else ""
        
        # 時間範囲
        alert_timestamps = [float(msg.timestamp) for msg in alert_messages]
        first_alert_time = None
        last_alert_time = None
        
        if alert_timestamps:
            first_alert_time = datetime.fromtimestamp(min(alert_timestamps)).isoformat()
            last_alert_time = datetime.fromtimestamp(max(alert_timestamps)).isoformat()
        
        # キーワード頻度分析
        keywords_freq = {}
        for msg in alert_messages:
            text_upper = msg.text.upper()
            for keyword in self.alert_keywords:
                if keyword.upper() in text_upper:
                    keywords_freq[keyword] = keywords_freq.get(keyword, 0) + 1
        
        return ChannelAnalysis(
            channel_name=channel_name,
            message_count=len(channel_messages),
            alert_count=len(alert_messages),
            alert_rate=round(alert_rate, 2),
            most_common_alert_level=most_common_level,
            first_alert_time=first_alert_time,
            last_alert_time=last_alert_time,
            keywords_frequency=keywords_freq
        )
    
    async def generate_alert_report(
        self, 
        hours_back: Optional[int] = None,
        include_channel_analysis: bool = True
    ) -> Dict[str, Any]:
        """包括的なアラートレポートを生成"""
        try:
            # アラートデータ取得
            messages = await self.get_recent_alerts(hours_back)
            
            # 全体分析
            summary = self.analyze_alert_patterns(messages)
            
            # レポート構造
            report = {
                "generated_at": datetime.now().isoformat(),
                "module": "slack_alert_reader",
                "config": {
                    "channels": self.channels,
                    "lookback_hours": hours_back or self.lookback_hours,
                    "max_messages_per_channel": self.max_messages
                },
                "summary": summary.to_dict(),
                "raw_messages": [
                    {
                        "channel": msg.channel_name,
                        "timestamp": msg.timestamp,
                        "text": msg.text,
                        "user": msg.user,
                        "alert_level": msg.alert_level,
                        "url": msg.message_url
                    }
                    for msg in messages[:50]  # 最新50件のみ
                ]
            }
            
            # チャンネル別分析を追加
            if include_channel_analysis:
                channel_analyses = []
                for channel in self.channels:
                    analysis = self.analyze_channel(channel, messages)
                    channel_analyses.append(analysis.to_dict())
                
                report["channel_analysis"] = channel_analyses
            
            self.logger.info(f"Alert report generated: {summary.total_alerts} alerts found")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating alert report: {e}")
            return {
                "generated_at": datetime.now().isoformat(),
                "module": "slack_alert_reader",
                "error": str(e),
                "summary": AlertSummary(0, 0, 0, 0, 0, [], {}, "").to_dict()
            }
    
    async def get_alert_context(self, message: SlackMessage) -> Dict[str, Any]:
        """特定アラートのコンテキスト情報を取得"""
        context = {
            "message": {
                "channel": message.channel_name,
                "timestamp": message.timestamp,
                "text": message.text,
                "user": message.user,
                "alert_level": message.alert_level,
                "url": message.message_url
            },
            "thread_replies": [],
            "reactions": message.reactions
        }
        
        # スレッドがある場合は返信も取得
        if message.thread_ts:
            try:
                replies = await self.slack_client.get_thread_replies(
                    message.channel, 
                    message.thread_ts
                )
                context["thread_replies"] = [
                    {
                        "timestamp": reply.timestamp,
                        "text": reply.text,
                        "user": reply.user
                    }
                    for reply in replies
                ]
            except Exception as e:
                self.logger.warning(f"Failed to get thread context: {e}")
        
        return context
    
    def export_alerts_json(self, messages: List[SlackMessage], filepath: str) -> bool:
        """アラートデータをJSONファイルに出力"""
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "module": "slack_alert_reader",
                "total_messages": len(messages),
                "messages": [
                    {
                        "channel": msg.channel_name,
                        "timestamp": msg.timestamp,
                        "datetime": msg.datetime.isoformat(),
                        "text": msg.text,
                        "user": msg.user,
                        "is_alert": msg.is_alert,
                        "alert_level": msg.alert_level,
                        "thread_ts": msg.thread_ts,
                        "reactions_count": len(msg.reactions),
                        "url": msg.message_url
                    }
                    for msg in messages
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Alerts exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export alerts: {e}")
            return False


# モジュール使用例
async def main():
    """使用例"""
    reader = SlackAlertReader()
    
    # セットアップ確認
    if not await reader.verify_setup():
        print("❌ Setup verification failed")
        return
    
    # アラートレポート生成
    report = await reader.generate_alert_report(hours_back=6)
    print(f"📊 Alert Report Generated")
    print(f"   Total alerts: {report['summary']['total_alerts']}")
    print(f"   Critical: {report['summary']['critical_count']}")
    print(f"   Warning: {report['summary']['warning_count']}")
    print(f"   Most active channel: {report['summary']['most_active_channel']}")


if __name__ == "__main__":
    asyncio.run(main())
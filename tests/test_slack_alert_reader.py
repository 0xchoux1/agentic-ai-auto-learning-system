"""
Test cases for Slack Alert Reader Module
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

import pytest

from aals.integrations.slack_client import SlackAPIClient, SlackMessage
from aals.modules.slack_alert_reader import SlackAlertReader, AlertSummary, ChannelAnalysis


class TestSlackMessage:
    """SlackMessageデータクラスのテスト"""
    
    def test_slack_message_creation(self):
        """SlackMessage作成のテスト"""
        msg = SlackMessage(
            channel="C1234567890",
            channel_name="alerts",
            timestamp="1672531200.123456",
            user="U1234567890",
            text="CRITICAL: Server is down",
            thread_ts="1672531200.123456"
        )
        
        assert msg.channel == "C1234567890"
        assert msg.channel_name == "alerts"
        assert msg.text == "CRITICAL: Server is down"
        assert msg.reactions == []  # デフォルト値
    
    def test_datetime_property(self):
        """datetime変換のテスト"""
        timestamp = "1672531200.123456"
        msg = SlackMessage(
            channel="C1234567890",
            channel_name="alerts",
            timestamp=timestamp,
            user="U1234567890",
            text="Test message"
        )
        
        expected_datetime = datetime.fromtimestamp(float(timestamp))
        assert msg.datetime == expected_datetime
    
    def test_message_url_property(self):
        """メッセージURL生成のテスト"""
        msg = SlackMessage(
            channel="C1234567890",
            channel_name="alerts",
            timestamp="1672531200.123456",
            user="U1234567890",
            text="Test message"
        )
        
        expected_url = "https://slack.com/app_redirect?channel=C1234567890&message_ts=1672531200123456"
        assert msg.message_url == expected_url


class TestSlackAPIClient:
    """SlackAPIClientのテスト"""
    
    @pytest.fixture
    def mock_slack_client(self):
        """モックされたSlackクライアント"""
        with patch('aals.integrations.slack_client.WebClient') as mock_web_client:
            # 環境変数でトークンを設定
            with patch.dict('os.environ', {'AALS_SLACK_TOKEN': 'xoxb-test-token'}):
                client = SlackAPIClient()
                yield client, mock_web_client
    
    @pytest.mark.asyncio
    async def test_verify_connection_success(self, mock_slack_client):
        """接続確認成功のテスト"""
        client, mock_web_client = mock_slack_client
        mock_web_client.return_value.auth_test.return_value = {
            "ok": True,
            "user": "test_bot"
        }
        
        result = await client.verify_connection()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_connection_failure(self, mock_slack_client):
        """接続確認失敗のテスト"""
        client, mock_web_client = mock_slack_client
        mock_web_client.return_value.auth_test.return_value = {
            "ok": False,
            "error": "invalid_auth"
        }
        
        result = await client.verify_connection()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_channel_id(self, mock_slack_client):
        """チャンネルID取得のテスト"""
        client, mock_web_client = mock_slack_client
        mock_web_client.return_value.conversations_list.return_value = {
            "ok": True,
            "channels": [
                {"id": "C1234567890", "name": "alerts"},
                {"id": "C0987654321", "name": "incidents"}
            ]
        }
        
        channel_id = await client.get_channel_id("alerts")
        assert channel_id == "C1234567890"
        
        # チャンネル名前に#がある場合
        channel_id = await client.get_channel_id("#incidents")
        assert channel_id == "C0987654321"
        
        # 存在しないチャンネル
        channel_id = await client.get_channel_id("nonexistent")
        assert channel_id is None
    
    def test_analyze_alert_message(self, mock_slack_client):
        """アラートメッセージ分析のテスト"""
        client, _ = mock_slack_client
        
        # CRITICALアラート
        is_alert, level = client._analyze_alert_message("CRITICAL: Server is down")
        assert is_alert is True
        assert level == "critical"
        
        # WARNINGアラート
        is_alert, level = client._analyze_alert_message("WARNING: High CPU usage")
        assert is_alert is True
        assert level == "warning"
        
        # ERRORアラート
        is_alert, level = client._analyze_alert_message("Error in application")
        assert is_alert is True
        assert level == "error"
        
        # 非アラートメッセージ
        is_alert, level = client._analyze_alert_message("Normal status update")
        assert is_alert is False
        assert level is None
        
        # パターンマッチング
        is_alert, level = client._analyze_alert_message("Prometheus alert firing")
        assert is_alert is True
        assert level == "info"


class TestSlackAlertReader:
    """SlackAlertReaderモジュールのテスト"""
    
    @pytest.fixture
    def mock_config(self):
        """モック設定"""
        with patch('aals.modules.slack_alert_reader.get_config_manager') as mock_manager:
            mock_config_manager = MagicMock()
            mock_config = MagicMock()
            mock_module_config = MagicMock()
            
            mock_config_manager.config = mock_config
            mock_config_manager.get_module_config.return_value = mock_module_config
            
            mock_module_config.enabled = True
            mock_module_config.config = {
                "channels": ["#alerts", "#incidents"],
                "max_messages": 100,
                "lookback_hours": 24,
                "alert_keywords": ["CRITICAL", "WARNING", "ERROR"]
            }
            
            mock_manager.return_value = mock_config_manager
            yield mock_config_manager
    
    @pytest.fixture
    def sample_messages(self) -> List[SlackMessage]:
        """サンプルメッセージデータ"""
        return [
            SlackMessage(
                channel="C1234567890",
                channel_name="alerts",
                timestamp="1672531200.123456",
                user="U1234567890",
                text="CRITICAL: Database server is down",
                is_alert=True,
                alert_level="critical"
            ),
            SlackMessage(
                channel="C1234567890",
                channel_name="alerts",
                timestamp="1672531100.123456",
                user="U1234567890",
                text="WARNING: High memory usage detected",
                is_alert=True,
                alert_level="warning"
            ),
            SlackMessage(
                channel="C0987654321",
                channel_name="incidents",
                timestamp="1672531000.123456",
                user="U1234567890",
                text="ERROR: Application failed to start",
                is_alert=True,
                alert_level="error"
            ),
            SlackMessage(
                channel="C1234567890",
                channel_name="alerts",
                timestamp="1672530900.123456",
                user="U1234567890",
                text="Regular status update",
                is_alert=False,
                alert_level=None
            )
        ]
    
    def test_initialization(self, mock_config):
        """初期化のテスト"""
        with patch('aals.modules.slack_alert_reader.SlackAPIClient'):
            reader = SlackAlertReader()
            assert reader.channels == ["#alerts", "#incidents"]
            assert reader.max_messages == 100
            assert reader.lookback_hours == 24
    
    def test_initialization_disabled_module(self, mock_config):
        """無効化されたモジュールの初期化テスト"""
        mock_config.get_module_config.return_value.enabled = False
        
        with patch('aals.modules.slack_alert_reader.SlackAPIClient'):
            with pytest.raises(RuntimeError, match="not enabled"):
                SlackAlertReader()
    
    def test_analyze_alert_patterns(self, mock_config, sample_messages):
        """アラートパターン分析のテスト"""
        with patch('aals.modules.slack_alert_reader.SlackAPIClient'):
            reader = SlackAlertReader()
            
            # アラートメッセージのみフィルタ
            alert_messages = [msg for msg in sample_messages if msg.is_alert]
            summary = reader.analyze_alert_patterns(alert_messages)
            
            assert summary.total_alerts == 3
            assert summary.critical_count == 1
            assert summary.warning_count == 1
            assert summary.error_count == 1
            assert summary.info_count == 0
            assert "alerts" in summary.channels
            assert "incidents" in summary.channels
            assert summary.most_active_channel == "alerts"  # 2つのアラートがある
    
    def test_analyze_alert_patterns_empty(self, mock_config):
        """空のメッセージリストでの分析テスト"""
        with patch('aals.modules.slack_alert_reader.SlackAPIClient'):
            reader = SlackAlertReader()
            summary = reader.analyze_alert_patterns([])
            
            assert summary.total_alerts == 0
            assert summary.critical_count == 0
            assert summary.warning_count == 0
            assert summary.error_count == 0
            assert summary.info_count == 0
            assert summary.channels == []
            assert summary.most_active_channel == ""
    
    def test_analyze_channel(self, mock_config, sample_messages):
        """チャンネル分析のテスト"""
        with patch('aals.modules.slack_alert_reader.SlackAPIClient'):
            reader = SlackAlertReader()
            
            analysis = reader.analyze_channel("alerts", sample_messages)
            
            assert analysis.channel_name == "alerts"
            assert analysis.message_count == 3  # alertsチャンネルのメッセージ数
            assert analysis.alert_count == 2   # alertsチャンネルのアラート数
            assert analysis.alert_rate == 66.67  # (2/3) * 100
            assert analysis.most_common_alert_level in ["critical", "warning"]
    
    def test_analyze_channel_nonexistent(self, mock_config, sample_messages):
        """存在しないチャンネルの分析テスト"""
        with patch('aals.modules.slack_alert_reader.SlackAPIClient'):
            reader = SlackAlertReader()
            
            analysis = reader.analyze_channel("nonexistent", sample_messages)
            
            assert analysis.channel_name == "nonexistent"
            assert analysis.message_count == 0
            assert analysis.alert_count == 0
            assert analysis.alert_rate == 0.0
    
    def test_export_alerts_json(self, mock_config, sample_messages):
        """JSONエクスポートのテスト"""
        with patch('aals.modules.slack_alert_reader.SlackAPIClient'):
            reader = SlackAlertReader()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                result = reader.export_alerts_json(sample_messages, temp_path)
                assert result is True
                
                # ファイル内容を確認
                with open(temp_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                assert data["module"] == "slack_alert_reader"
                assert data["total_messages"] == len(sample_messages)
                assert len(data["messages"]) == len(sample_messages)
                assert "exported_at" in data
                
                # 最初のメッセージの内容確認
                first_msg = data["messages"][0]
                assert first_msg["channel"] == "alerts"
                assert first_msg["text"] == "CRITICAL: Database server is down"
                assert first_msg["alert_level"] == "critical"
                
            finally:
                import os
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_verify_setup_success(self, mock_config):
        """セットアップ確認成功のテスト"""
        with patch('aals.modules.slack_alert_reader.SlackAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.verify_connection.return_value = True
            mock_client.get_channel_id.return_value = "C1234567890"
            mock_client_class.return_value = mock_client
            
            reader = SlackAlertReader()
            result = await reader.verify_setup()
            
            assert result is True
            mock_client.verify_connection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_verify_setup_missing_channel(self, mock_config):
        """チャンネル未発見でのセットアップ確認テスト"""
        with patch('aals.modules.slack_alert_reader.SlackAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.verify_connection.return_value = True
            mock_client.get_channel_id.return_value = None  # チャンネルが見つからない
            mock_client_class.return_value = mock_client
            
            reader = SlackAlertReader()
            result = await reader.verify_setup()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_recent_alerts(self, mock_config, sample_messages):
        """最近のアラート取得のテスト"""
        with patch('aals.modules.slack_alert_reader.SlackAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_all_alert_messages.return_value = sample_messages
            mock_client_class.return_value = mock_client
            
            reader = SlackAlertReader()
            alerts = await reader.get_recent_alerts(hours_back=12)
            
            assert len(alerts) == len(sample_messages)
            mock_client.get_all_alert_messages.assert_called_once_with(
                channels=["#alerts", "#incidents"],
                hours_back=12,
                max_messages_per_channel=100
            )
    
    @pytest.mark.asyncio
    async def test_generate_alert_report(self, mock_config, sample_messages):
        """アラートレポート生成のテスト"""
        with patch('aals.modules.slack_alert_reader.SlackAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            # アラートメッセージのみ返す
            alert_messages = [msg for msg in sample_messages if msg.is_alert]
            mock_client.get_all_alert_messages.return_value = alert_messages
            mock_client_class.return_value = mock_client
            
            reader = SlackAlertReader()
            report = await reader.generate_alert_report(hours_back=6)
            
            assert report["module"] == "slack_alert_reader"
            assert "generated_at" in report
            assert "summary" in report
            assert "channel_analysis" in report
            assert "raw_messages" in report
            
            # サマリーの確認
            summary = report["summary"]
            assert summary["total_alerts"] == 3
            assert summary["critical_count"] == 1
            assert summary["warning_count"] == 1
            assert summary["error_count"] == 1
            
            # 設定の確認
            config = report["config"]
            assert config["channels"] == ["#alerts", "#incidents"]
            assert config["lookback_hours"] == 6


if __name__ == "__main__":
    pytest.main([__file__])
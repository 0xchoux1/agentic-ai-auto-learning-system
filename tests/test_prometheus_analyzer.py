#!/usr/bin/env python3
"""
AALS Module 4: Prometheus Analyzer テスト
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from aals.modules.prometheus_analyzer import (
    PrometheusAnalyzer,
    MetricThreshold,
    AlertEvent,
    AlertSeverity,
    MetricAnalysis,
    SystemHealthReport
)
from aals.integrations.prometheus_client import (
    PrometheusAPIClient,
    PrometheusMetric,
    MetricRange,
    PrometheusQueryResult
)


@pytest.fixture
def sample_prometheus_metric():
    """サンプルPrometheusメトリクス"""
    return PrometheusMetric(
        metric_name="cpu_usage_percent",
        labels={"instance": "server01", "job": "node"},
        timestamp=datetime.now(),
        value=75.5
    )


@pytest.fixture
def sample_metric_range():
    """サンプルメトリクス範囲データ"""
    now = datetime.now()
    values = []
    for i in range(60):  # 60データポイント
        timestamp = now - timedelta(minutes=60-i)
        value = 50.0 + (i * 0.5)  # 徐々に増加するパターン
        values.append((timestamp, value))
    
    return MetricRange(
        metric_name="cpu_usage_percent",
        labels={"instance": "server01"},
        values=values
    )


@pytest.fixture
def prometheus_analyzer():
    """Prometheus Analyzer インスタンス"""
    with patch('aals.modules.prometheus_analyzer.get_config_manager') as mock_config:
        mock_module_config = MagicMock()
        mock_module_config.enabled = True
        mock_module_config.config = {
            "url": "http://localhost:9090",
            "timeout": 30,
            "thresholds": {
                "cpu_warning": 70.0,
                "cpu_critical": 90.0,
                "memory_warning": 80.0,
                "memory_critical": 95.0
            }
        }
        mock_config.return_value.get_module_config.return_value = mock_module_config
        
        analyzer = PrometheusAnalyzer()
        return analyzer


class TestPrometheusAPIClient:
    """PrometheusAPIClient テスト"""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """クライアント初期化テスト"""
        client = PrometheusAPIClient("http://localhost:9090", timeout=30)
        assert client.base_url == "http://localhost:9090"
        assert client.timeout == 30
    
    @pytest.mark.asyncio
    async def test_verify_connection_success(self):
        """接続確認成功テスト"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = PrometheusAPIClient("http://localhost:9090")
            
            async with client:
                result = await client.verify_connection()
                assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_connection_failure(self):
        """接続確認失敗テスト"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = PrometheusAPIClient("http://localhost:9090")
            
            async with client:
                result = await client.verify_connection()
                assert result is False
    
    @pytest.mark.asyncio
    async def test_query_instant_success(self):
        """即座クエリ成功テスト"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "success",
                "data": {
                    "resultType": "vector",
                    "result": []
                }
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = PrometheusAPIClient("http://localhost:9090")
            
            async with client:
                result = await client.query_instant("up")
                assert isinstance(result, PrometheusQueryResult)
                assert result.status == "success"
    
    def test_parse_instant_result(self):
        """即座クエリ結果パーステスト"""
        client = PrometheusAPIClient("http://localhost:9090")
        
        result = PrometheusQueryResult(
            status="success",
            data={
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "cpu_usage", "instance": "server01"},
                        "value": [1640995200, "75.5"]
                    }
                ]
            }
        )
        
        metrics = client.parse_instant_result(result)
        assert len(metrics) == 1
        assert metrics[0].metric_name == "cpu_usage"
        assert metrics[0].value == 75.5
        assert metrics[0].labels["instance"] == "server01"
    
    def test_parse_range_result(self):
        """範囲クエリ結果パーステスト"""
        client = PrometheusAPIClient("http://localhost:9090")
        
        result = PrometheusQueryResult(
            status="success",
            data={
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "cpu_usage", "instance": "server01"},
                        "values": [
                            [1640995200, "75.0"],
                            [1640995260, "76.0"]
                        ]
                    }
                ]
            }
        )
        
        ranges = client.parse_range_result(result)
        assert len(ranges) == 1
        assert ranges[0].metric_name == "cpu_usage"
        assert len(ranges[0].values) == 2
        assert ranges[0].latest_value == 76.0


class TestPrometheusAnalyzer:
    """PrometheusAnalyzer テスト"""
    
    def test_initialization(self, prometheus_analyzer):
        """初期化テスト"""
        assert prometheus_analyzer.prometheus_url == "http://localhost:9090"
        assert prometheus_analyzer.timeout == 30
        assert len(prometheus_analyzer.thresholds) > 0
    
    def test_load_default_thresholds(self, prometheus_analyzer):
        """既定閾値読み込みテスト"""
        thresholds = prometheus_analyzer.thresholds
        
        # CPU閾値チェック
        cpu_threshold = next((t for t in thresholds if t.metric_name == "cpu_usage_percent"), None)
        assert cpu_threshold is not None
        assert cpu_threshold.warning_threshold == 70.0
        assert cpu_threshold.critical_threshold == 90.0
    
    @pytest.mark.asyncio
    async def test_verify_setup_success(self, prometheus_analyzer):
        """セットアップ確認成功テスト"""
        with patch('aals.modules.prometheus_analyzer.PrometheusAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.verify_connection.return_value = True
            mock_client_class.return_value = mock_client
            
            result = await prometheus_analyzer.verify_setup()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_setup_failure(self, prometheus_analyzer):
        """セットアップ確認失敗テスト"""
        with patch('aals.modules.prometheus_analyzer.PrometheusAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.verify_connection.return_value = False
            mock_client_class.return_value = mock_client
            
            result = await prometheus_analyzer.verify_setup()
            assert result is False
    
    def test_analyze_metric_trend_increasing(self, prometheus_analyzer, sample_metric_range):
        """メトリクストレンド分析（増加）テスト"""
        trend, score = prometheus_analyzer.analyze_metric_trend(sample_metric_range)
        assert trend == "increasing"
        assert score > 0
    
    def test_analyze_metric_trend_stable(self, prometheus_analyzer):
        """メトリクストレンド分析（安定）テスト"""
        now = datetime.now()
        stable_values = []
        for i in range(10):
            timestamp = now - timedelta(minutes=10-i)
            value = 50.0  # 一定値
            stable_values.append((timestamp, value))
        
        stable_range = MetricRange(
            metric_name="test_metric",
            labels={},
            values=stable_values
        )
        
        trend, score = prometheus_analyzer.analyze_metric_trend(stable_range)
        assert trend == "stable"
        assert score < 0.1
    
    def test_detect_anomalies(self, prometheus_analyzer):
        """異常検知テスト"""
        now = datetime.now()
        values = []
        
        # 正常値（平均50、標準偏差約5）
        normal_values = [45, 47, 49, 51, 53, 48, 52, 50, 46, 54]
        for i, val in enumerate(normal_values):
            timestamp = now - timedelta(minutes=len(normal_values)-i)
            values.append((timestamp, float(val)))
        
        # 異常値を追加
        values.append((now, 80.0))  # 明らかな異常値
        
        anomaly_range = MetricRange(
            metric_name="test_metric",
            labels={},
            values=values
        )
        
        anomaly_score = prometheus_analyzer.detect_anomalies(anomaly_range)
        assert anomaly_score > 0.5  # 高い異常スコア
    
    def test_check_thresholds_warning(self, prometheus_analyzer, sample_prometheus_metric):
        """閾値チェック（警告）テスト"""
        # Warning閾値（70.0）を超える値（75.5）
        alerts = prometheus_analyzer.check_thresholds(sample_prometheus_metric)
        
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert alerts[0].current_value == 75.5
        assert alerts[0].threshold_value == 70.0
    
    def test_check_thresholds_critical(self, prometheus_analyzer):
        """閾値チェック（クリティカル）テスト"""
        critical_metric = PrometheusMetric(
            metric_name="cpu_usage_percent",
            labels={"instance": "server01"},
            timestamp=datetime.now(),
            value=95.0  # Critical閾値（90.0）を超える
        )
        
        alerts = prometheus_analyzer.check_thresholds(critical_metric)
        
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert alerts[0].current_value == 95.0
        assert alerts[0].threshold_value == 90.0
    
    def test_check_thresholds_no_alert(self, prometheus_analyzer):
        """閾値チェック（アラートなし）テスト"""
        normal_metric = PrometheusMetric(
            metric_name="cpu_usage_percent",
            labels={"instance": "server01"},
            timestamp=datetime.now(),
            value=50.0  # 正常範囲
        )
        
        alerts = prometheus_analyzer.check_thresholds(normal_metric)
        assert len(alerts) == 0
    
    @pytest.mark.asyncio
    async def test_get_current_metrics(self, prometheus_analyzer):
        """現在メトリクス取得テスト"""
        with patch('aals.modules.prometheus_analyzer.PrometheusAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            
            # モックレスポンス設定
            mock_result = PrometheusQueryResult(
                status="success",
                data={
                    "resultType": "vector",
                    "result": [
                        {
                            "metric": {"__name__": "cpu_usage", "instance": "server01"},
                            "value": [1640995200, "75.5"]
                        }
                    ]
                }
            )
            
            mock_client.query_instant.return_value = mock_result
            mock_client.parse_instant_result.return_value = [
                PrometheusMetric(
                    metric_name="cpu_usage",
                    labels={"instance": "server01"},
                    timestamp=datetime.now(),
                    value=75.5
                )
            ]
            
            mock_client_class.return_value = mock_client
            
            metrics = await prometheus_analyzer.get_current_metrics(["cpu_usage"])
            
            assert len(metrics) == 1
            assert metrics[0].metric_name == "cpu_usage"
            assert metrics[0].value == 75.5
    
    @pytest.mark.asyncio
    async def test_get_metric_history(self, prometheus_analyzer):
        """メトリクス履歴取得テスト"""
        with patch('aals.modules.prometheus_analyzer.PrometheusAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            
            # モック履歴データ
            now = datetime.now()
            values = [(now - timedelta(minutes=i), 50.0 + i) for i in range(10)]
            
            mock_range = MetricRange(
                metric_name="cpu_usage",
                labels={"instance": "server01"},
                values=values
            )
            
            mock_result = PrometheusQueryResult(status="success", data={})
            mock_client.query_range.return_value = mock_result
            mock_client.parse_range_result.return_value = [mock_range]
            
            mock_client_class.return_value = mock_client
            
            history = await prometheus_analyzer.get_metric_history("cpu_usage", hours_back=1)
            
            assert history is not None
            assert history.metric_name == "cpu_usage"
            assert len(history.values) == 10
    
    def test_generate_recommendations(self, prometheus_analyzer):
        """推奨事項生成テスト"""
        # テスト用の分析データ
        analyses = [
            MetricAnalysis(
                metric_name="cpu_usage",
                current_value=75.0,
                trend="increasing",
                anomaly_score=0.8,  # 高い異常スコア
                alerts=[],
                statistics={},
                labels={}
            )
        ]
        
        alerts = [
            AlertEvent(
                metric_name="memory_usage",
                current_value=96.0,
                threshold_value=95.0,
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.now(),
                labels={},
                message="Critical alert",
                comparison=">"
            )
        ]
        
        recommendations = prometheus_analyzer._generate_recommendations(analyses, alerts)
        
        assert len(recommendations) > 0
        assert any("Critical alerts detected" in rec for rec in recommendations)
        assert any("High anomaly detected" in rec for rec in recommendations)


class TestMetricRange:
    """MetricRange テスト"""
    
    def test_metric_range_properties(self, sample_metric_range):
        """MetricRange プロパティテスト"""
        assert sample_metric_range.latest_value == 79.5  # 最後の値
        assert sample_metric_range.min_value == 50.0
        assert sample_metric_range.max_value == 79.5
        assert abs(sample_metric_range.avg_value - 64.75) < 0.1
    
    def test_empty_metric_range(self):
        """空のMetricRange テスト"""
        empty_range = MetricRange(
            metric_name="test",
            labels={},
            values=[]
        )
        
        assert empty_range.latest_value is None
        assert empty_range.min_value is None
        assert empty_range.max_value is None
        assert empty_range.avg_value is None


@pytest.mark.asyncio
async def test_full_integration(prometheus_analyzer):
    """完全統合テスト"""
    with patch('aals.integrations.prometheus_client.PrometheusAPIClient') as mock_client_class:
        mock_client = AsyncMock()
        
        # 現在値のモック
        current_metrics = [
            PrometheusMetric(
                metric_name="cpu_usage_percent",
                labels={"instance": "server01"},
                timestamp=datetime.now(),
                value=85.0  # Warning閾値超過
            )
        ]
        
        # 履歴データのモック
        now = datetime.now()
        history_values = [(now - timedelta(minutes=i), 80.0 + i) for i in range(10)]
        history_range = MetricRange(
            metric_name="cpu_usage_percent",
            labels={"instance": "server01"},
            values=history_values
        )
        
        # モック設定
        mock_result = PrometheusQueryResult(status="success", data={})
        mock_client.query_instant.return_value = mock_result
        mock_client.parse_instant_result.return_value = current_metrics
        mock_client.query_range.return_value = mock_result
        mock_client.parse_range_result.return_value = [history_range]
        
        mock_client_class.return_value = mock_client
        
        # システムヘルス分析実行
        with patch.object(prometheus_analyzer, 'get_current_metrics', return_value=current_metrics):
            with patch.object(prometheus_analyzer, 'get_metric_history', return_value=history_range):
                report = await prometheus_analyzer.analyze_system_health(hours_back=1)
        
        # 結果検証
        assert isinstance(report, SystemHealthReport)
        assert report.overall_health in ["healthy", "warning", "critical"]
        assert report.total_metrics > 0
        assert len(report.metric_analyses) > 0
        assert len(report.recommendations) > 0
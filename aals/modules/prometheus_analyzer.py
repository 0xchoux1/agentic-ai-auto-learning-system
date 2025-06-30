#!/usr/bin/env python3
"""
AALS Module 4: Prometheus Analyzer
メトリクス収集・分析・異常検知モジュール
"""

import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from aals.core.config import get_config_manager
from aals.core.logger import get_logger, AuditAction, AuditLogEntry, audit_log
from aals.integrations.prometheus_client import (
    PrometheusAPIClient, 
    PrometheusMetric, 
    MetricRange
)


logger = get_logger(__name__)


class AlertSeverity(Enum):
    """アラート重要度"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class MetricThreshold:
    """メトリクス閾値定義"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = ">"  # >, <, >=, <=
    labels_filter: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertEvent:
    """アラートイベント"""
    metric_name: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    timestamp: datetime
    labels: Dict[str, str]
    message: str
    comparison: str


@dataclass
class MetricAnalysis:
    """メトリクス分析結果"""
    metric_name: str
    current_value: Optional[float]
    trend: str  # "increasing", "decreasing", "stable"
    anomaly_score: float  # 0.0-1.0
    alerts: List[AlertEvent]
    statistics: Dict[str, float]
    labels: Dict[str, str]


@dataclass
class SystemHealthReport:
    """システムヘルス総合レポート"""
    timestamp: datetime
    overall_health: str  # "healthy", "warning", "critical"
    total_metrics: int
    alerts_count: Dict[AlertSeverity, int]
    metric_analyses: List[MetricAnalysis]
    recommendations: List[str]


class PrometheusAnalyzer:
    """Prometheusメトリクス分析器"""
    
    def __init__(self):
        """初期化"""
        self.config_manager = get_config_manager()
        self.config = self.config_manager.get_module_config("prometheus_analyzer")
        
        # Prometheus接続設定
        prometheus_config = self.config.config
        self.prometheus_url = prometheus_config.get("url", "http://localhost:9090")
        self.timeout = prometheus_config.get("timeout", 30)
        
        # 既定の閾値設定
        self._load_default_thresholds()
        
        logger.info("Prometheus Analyzer initialized", 
                   prometheus_url=self.prometheus_url,
                   thresholds_count=len(self.thresholds))
    
    def _load_default_thresholds(self):
        """既定閾値設定を読み込み"""
        threshold_config = self.config.config.get("thresholds", {})
        
        self.thresholds = [
            # CPU使用率
            MetricThreshold(
                metric_name="cpu_usage_percent",
                warning_threshold=threshold_config.get("cpu_warning", 70.0),
                critical_threshold=threshold_config.get("cpu_critical", 90.0),
                comparison=">"
            ),
            # メモリ使用率
            MetricThreshold(
                metric_name="memory_usage_percent", 
                warning_threshold=threshold_config.get("memory_warning", 80.0),
                critical_threshold=threshold_config.get("memory_critical", 95.0),
                comparison=">"
            ),
            # ディスク使用率
            MetricThreshold(
                metric_name="disk_usage_percent",
                warning_threshold=threshold_config.get("disk_warning", 85.0),
                critical_threshold=threshold_config.get("disk_critical", 95.0),
                comparison=">"
            ),
            # ロードアベレージ
            MetricThreshold(
                metric_name="load_average_1m",
                warning_threshold=threshold_config.get("load_warning", 4.0),
                critical_threshold=threshold_config.get("load_critical", 8.0),
                comparison=">"
            ),
            # レスポンス時間
            MetricThreshold(
                metric_name="http_response_time_seconds",
                warning_threshold=threshold_config.get("response_warning", 2.0),
                critical_threshold=threshold_config.get("response_critical", 5.0),
                comparison=">"
            )
        ]
    
    async def verify_setup(self) -> bool:
        """セットアップ確認"""
        try:
            async with PrometheusAPIClient(self.prometheus_url, self.timeout) as client:
                connected = await client.verify_connection()
                
                if connected:
                    logger.info("Prometheus Analyzer setup verification completed successfully")
                    return True
                else:
                    logger.error("Prometheus Analyzer setup verification failed - connection failed")
                    return False
                    
        except Exception as e:
            logger.error("Prometheus Analyzer setup verification failed", 
                        error=str(e), exception_type=type(e).__name__)
            return False
    
    async def get_current_metrics(self, metric_queries: List[str]) -> List[PrometheusMetric]:
        """現在のメトリクス値を取得"""
        metrics = []
        
        async with PrometheusAPIClient(self.prometheus_url, self.timeout) as client:
            for query in metric_queries:
                try:
                    result = await client.query_instant(query)
                    parsed_metrics = client.parse_instant_result(result)
                    metrics.extend(parsed_metrics)
                    
                except Exception as e:
                    logger.error("Failed to get current metric", 
                                query=query, error=str(e))
        
        logger.info("Current metrics retrieved", 
                   queries_count=len(metric_queries), 
                   metrics_count=len(metrics))
        return metrics
    
    async def get_metric_history(
        self, 
        query: str, 
        hours_back: int = 1,
        step: str = "1m"
    ) -> Optional[MetricRange]:
        """メトリクス履歴を取得"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        async with PrometheusAPIClient(self.prometheus_url, self.timeout) as client:
            try:
                result = await client.query_range(query, start_time, end_time, step)
                ranges = client.parse_range_result(result)
                
                if ranges:
                    logger.info("Metric history retrieved", 
                               query=query, hours_back=hours_back,
                               data_points=len(ranges[0].values))
                    return ranges[0]
                else:
                    logger.warning("No metric history data found", query=query)
                    return None
                    
            except Exception as e:
                logger.error("Failed to get metric history", 
                            query=query, error=str(e))
                return None
    
    def analyze_metric_trend(self, metric_range: MetricRange) -> Tuple[str, float]:
        """メトリクストレンド分析"""
        if not metric_range.values or len(metric_range.values) < 3:
            return "unknown", 0.0
        
        values = [v[1] for v in metric_range.values]
        
        # 線形回帰で傾向を判定
        n = len(values)
        x_values = list(range(n))
        
        # 傾きを計算
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # 傾向を判定
        if abs(slope) < 0.01:  # 変化が小さい
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        # 変化率を計算（異常スコア的に使用）
        if len(values) > 1:
            recent_avg = statistics.mean(values[-min(5, len(values)):])
            older_avg = statistics.mean(values[:min(5, len(values))])
            if older_avg != 0:
                change_rate = abs((recent_avg - older_avg) / older_avg)
            else:
                change_rate = 0.0
        else:
            change_rate = 0.0
        
        return trend, min(change_rate, 1.0)
    
    def detect_anomalies(self, metric_range: MetricRange) -> float:
        """異常検知（シンプルな統計ベース）"""
        if not metric_range.values or len(metric_range.values) < 10:
            return 0.0
        
        values = [v[1] for v in metric_range.values]
        
        try:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
            
            if stdev == 0:
                return 0.0
            
            # 最新値のZ-scoreを計算
            latest_value = values[-1]
            z_score = abs((latest_value - mean) / stdev)
            
            # Z-scoreを0-1の異常スコアに変換
            # Z-score 2以上を異常とする
            anomaly_score = min(z_score / 2.0, 1.0)
            
            return anomaly_score
            
        except statistics.StatisticsError:
            return 0.0
    
    def check_thresholds(self, metric: PrometheusMetric) -> List[AlertEvent]:
        """閾値チェック"""
        alerts = []
        
        for threshold in self.thresholds:
            # メトリクス名とラベルフィルターをチェック
            if threshold.metric_name != metric.metric_name:
                continue
            
            # ラベルフィルターチェック
            if threshold.labels_filter:
                match = all(
                    metric.labels.get(k) == v 
                    for k, v in threshold.labels_filter.items()
                )
                if not match:
                    continue
            
            # 閾値チェック
            value = metric.value
            comp = threshold.comparison
            
            critical_exceeded = False
            warning_exceeded = False
            
            if comp == ">":
                critical_exceeded = value > threshold.critical_threshold
                warning_exceeded = value > threshold.warning_threshold
            elif comp == "<":
                critical_exceeded = value < threshold.critical_threshold
                warning_exceeded = value < threshold.warning_threshold
            elif comp == ">=":
                critical_exceeded = value >= threshold.critical_threshold
                warning_exceeded = value >= threshold.warning_threshold
            elif comp == "<=":
                critical_exceeded = value <= threshold.critical_threshold
                warning_exceeded = value <= threshold.warning_threshold
            
            # アラート生成
            if critical_exceeded:
                alerts.append(AlertEvent(
                    metric_name=metric.metric_name,
                    current_value=value,
                    threshold_value=threshold.critical_threshold,
                    severity=AlertSeverity.CRITICAL,
                    timestamp=metric.timestamp,
                    labels=metric.labels,
                    message=f"{metric.metric_name} is {comp} critical threshold",
                    comparison=comp
                ))
            elif warning_exceeded:
                alerts.append(AlertEvent(
                    metric_name=metric.metric_name,
                    current_value=value,
                    threshold_value=threshold.warning_threshold,
                    severity=AlertSeverity.WARNING,
                    timestamp=metric.timestamp,
                    labels=metric.labels,
                    message=f"{metric.metric_name} is {comp} warning threshold",
                    comparison=comp
                ))
        
        return alerts
    
    async def analyze_system_health(self, hours_back: int = 1) -> SystemHealthReport:
        """システムヘルス総合分析"""
        logger.info("Starting system health analysis", hours_back=hours_back)
        
        # 主要メトリクスクエリ
        key_queries = [
            "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",  # CPU使用率
            "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",  # メモリ使用率
            "(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100",  # ディスク使用率
            "node_load1",  # ロードアベレージ
            "up"  # サービス稼働状況
        ]
        
        analyses = []
        all_alerts = []
        
        # 現在値取得・分析
        current_metrics = await self.get_current_metrics(key_queries)
        
        for metric in current_metrics:
            # 履歴データ取得
            history = await self.get_metric_history(
                f"{metric.metric_name}{{{','.join([f'{k}=\"{v}\"' for k, v in metric.labels.items()])}}}"
            )
            
            # トレンド分析
            trend = "unknown"
            anomaly_score = 0.0
            stats = {}
            
            if history:
                trend, trend_score = self.analyze_metric_trend(history)
                anomaly_score = max(self.detect_anomalies(history), trend_score)
                
                values = [v[1] for v in history.values]
                if values:
                    stats = {
                        "min": min(values),
                        "max": max(values),
                        "avg": statistics.mean(values),
                        "latest": values[-1]
                    }
            
            # 閾値チェック
            metric_alerts = self.check_thresholds(metric)
            all_alerts.extend(metric_alerts)
            
            # 分析結果作成
            analysis = MetricAnalysis(
                metric_name=metric.metric_name,
                current_value=metric.value,
                trend=trend,
                anomaly_score=anomaly_score,
                alerts=metric_alerts,
                statistics=stats,
                labels=metric.labels
            )
            analyses.append(analysis)
        
        # アラート集計
        alerts_count = {
            AlertSeverity.CRITICAL: len([a for a in all_alerts if a.severity == AlertSeverity.CRITICAL]),
            AlertSeverity.WARNING: len([a for a in all_alerts if a.severity == AlertSeverity.WARNING]),
            AlertSeverity.INFO: len([a for a in all_alerts if a.severity == AlertSeverity.INFO])
        }
        
        # 総合ヘルス判定
        if alerts_count[AlertSeverity.CRITICAL] > 0:
            overall_health = "critical"
        elif alerts_count[AlertSeverity.WARNING] > 0:
            overall_health = "warning"
        else:
            overall_health = "healthy"
        
        # 推奨事項生成
        recommendations = self._generate_recommendations(analyses, all_alerts)
        
        report = SystemHealthReport(
            timestamp=datetime.now(),
            overall_health=overall_health,
            total_metrics=len(analyses),
            alerts_count=alerts_count,
            metric_analyses=analyses,
            recommendations=recommendations
        )
        
        # 監査ログ記録
        audit_log(AuditLogEntry(
            action=AuditAction.VIEW,
            resource="prometheus_metrics",
            result="success",
            details=f"システムヘルス分析完了 - {overall_health}状態, {len(all_alerts)}件のアラート",
            risk_level="low" if overall_health == "healthy" else "medium"
        ))
        
        logger.info("System health analysis completed", 
                   overall_health=overall_health,
                   total_metrics=len(analyses),
                   total_alerts=len(all_alerts))
        
        return report
    
    def _generate_recommendations(self, analyses: List[MetricAnalysis], alerts: List[AlertEvent]) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        # アラート基準の推奨
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            recommendations.append("Critical alerts detected - immediate investigation required")
        
        # 異常スコア基準の推奨
        high_anomaly_metrics = [a for a in analyses if a.anomaly_score > 0.7]
        if high_anomaly_metrics:
            recommendations.append(f"High anomaly detected in {len(high_anomaly_metrics)} metrics")
        
        # トレンド基準の推奨
        increasing_metrics = [a for a in analyses if a.trend == "increasing" and a.anomaly_score > 0.5]
        if increasing_metrics:
            recommendations.append(f"Monitor {len(increasing_metrics)} rapidly increasing metrics")
        
        # 一般的な推奨事項
        if not recommendations:
            recommendations.append("System appears healthy - continue monitoring")
        
        return recommendations
#!/usr/bin/env python3
"""
AALS Module 4: Prometheus Analyzer デモ
メトリクス分析機能の動作確認
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from aals.modules.prometheus_analyzer import PrometheusAnalyzer, SystemHealthReport
from aals.integrations.prometheus_client import PrometheusMetric, MetricRange
from aals.core.logger import get_logger


logger = get_logger(__name__)


async def demo_prometheus_analyzer():
    """Prometheus Analyzer デモ実行"""
    print("🔬 AALS Module 4: Prometheus Analyzer デモ")
    print("=" * 50)
    
    # 1. モジュール初期化
    print("\n📋 1. Prometheus Analyzer 初期化")
    print("-" * 30)
    
    analyzer = PrometheusAnalyzer()
    print(f"✅ 初期化完了: {analyzer.prometheus_url}")
    print(f"✅ 閾値設定数: {len(analyzer.thresholds)}個")
    
    # 2. セットアップ確認（モック）
    print(f"\n🔗 2. セットアップ確認")
    print("-" * 30)
    
    with patch('aals.modules.prometheus_analyzer.PrometheusAPIClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.verify_connection.return_value = True
        mock_client_class.return_value = mock_client
        
        setup_ok = await analyzer.verify_setup()
        print(f"✅ セットアップ確認: {'成功' if setup_ok else '失敗'}")
    
    # 3. サンプルメトリクス生成・分析
    print(f"\n📊 3. メトリクス分析機能テスト")
    print("-" * 30)
    
    # サンプルメトリクス
    now = datetime.now()
    sample_metrics = [
        PrometheusMetric(
            metric_name="cpu_usage_percent",
            labels={"instance": "web-server-01", "job": "node"},
            timestamp=now,
            value=85.0  # Warning閾値超過
        ),
        PrometheusMetric(
            metric_name="memory_usage_percent", 
            labels={"instance": "web-server-01", "job": "node"},
            timestamp=now,
            value=45.0  # 正常範囲
        ),
        PrometheusMetric(
            metric_name="disk_usage_percent",
            labels={"instance": "web-server-01", "job": "node", "mountpoint": "/"},
            timestamp=now,
            value=97.0  # Critical閾値超過
        )
    ]
    
    print(f"✅ サンプルメトリクス生成: {len(sample_metrics)}個")
    
    # 4. 閾値チェック
    print(f"\n⚠️  4. 閾値チェック・アラート検知")
    print("-" * 30)
    
    all_alerts = []
    for metric in sample_metrics:
        alerts = analyzer.check_thresholds(metric)
        all_alerts.extend(alerts)
        
        if alerts:
            alert = alerts[0]
            severity_emoji = "🚨" if alert.severity.value == "critical" else "⚠️"
            print(f"{severity_emoji} {alert.metric_name}: {alert.current_value:.1f} > {alert.threshold_value:.1f} ({alert.severity.value.upper()})")
        else:
            print(f"✅ {metric.metric_name}: {metric.value:.1f} (正常)")
    
    print(f"\n📈 アラート合計: {len(all_alerts)}件")
    
    # 5. トレンド分析デモ
    print(f"\n📈 5. トレンド分析デモ")
    print("-" * 30)
    
    # 増加傾向のサンプルデータ
    increasing_values = []
    for i in range(20):
        timestamp = now - timedelta(minutes=20-i)
        value = 60.0 + (i * 1.5)  # 徐々に増加
        increasing_values.append((timestamp, value))
    
    increasing_range = MetricRange(
        metric_name="cpu_usage_percent",
        labels={"instance": "web-server-01"},
        values=increasing_values
    )
    
    trend, anomaly_score = analyzer.analyze_metric_trend(increasing_range)
    print(f"✅ トレンド分析: {trend} (異常スコア: {anomaly_score:.2f})")
    print(f"✅ 最新値: {increasing_range.latest_value:.1f}%")
    print(f"✅ 平均値: {increasing_range.avg_value:.1f}%")
    
    # 6. 異常検知デモ
    print(f"\n🔍 6. 異常検知デモ")
    print("-" * 30)
    
    # 正常値＋異常値のパターン
    anomaly_values = []
    # 正常値（平均50、標準偏差約5）
    normal_pattern = [48, 52, 49, 51, 50, 47, 53, 49, 52, 48]
    for i, val in enumerate(normal_pattern):
        timestamp = now - timedelta(minutes=len(normal_pattern)-i)
        anomaly_values.append((timestamp, float(val)))
    
    # 異常値を追加
    anomaly_values.append((now, 95.0))  # 明らかな異常値
    
    anomaly_range = MetricRange(
        metric_name="response_time_seconds",
        labels={"service": "api"},
        values=anomaly_values
    )
    
    anomaly_score = analyzer.detect_anomalies(anomaly_range)
    print(f"✅ 異常検知: スコア {anomaly_score:.2f} ({'高' if anomaly_score > 0.5 else '低'}異常度)")
    print(f"✅ 最新値: {anomaly_range.latest_value:.1f} (正常範囲: 47-53)")
    
    # 7. 推奨事項生成
    print(f"\n💡 7. 推奨事項生成")
    print("-" * 30)
    
    # 分析データのサンプル作成
    from aals.modules.prometheus_analyzer import MetricAnalysis
    
    sample_analyses = [
        MetricAnalysis(
            metric_name="cpu_usage_percent",
            current_value=85.0,
            trend="increasing", 
            anomaly_score=0.3,
            alerts=all_alerts[:1] if all_alerts else [],
            statistics={"min": 70.0, "max": 85.0, "avg": 77.5},
            labels={"instance": "web-server-01"}
        ),
        MetricAnalysis(
            metric_name="disk_usage_percent",
            current_value=97.0,
            trend="stable",
            anomaly_score=0.8,
            alerts=all_alerts[1:] if len(all_alerts) > 1 else [],
            statistics={"min": 95.0, "max": 97.0, "avg": 96.0},
            labels={"instance": "web-server-01"}
        )
    ]
    
    recommendations = analyzer._generate_recommendations(sample_analyses, all_alerts)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"💡 推奨 {i}: {rec}")
    
    # 8. システムヘルス総合判定
    print(f"\n🎯 8. システムヘルス総合判定")
    print("-" * 30)
    
    # アラート集計
    from aals.modules.prometheus_analyzer import AlertSeverity
    
    alerts_count = {
        AlertSeverity.CRITICAL: len([a for a in all_alerts if a.severity == AlertSeverity.CRITICAL]),
        AlertSeverity.WARNING: len([a for a in all_alerts if a.severity == AlertSeverity.WARNING]),
        AlertSeverity.INFO: 0
    }
    
    # 総合ヘルス判定
    if alerts_count[AlertSeverity.CRITICAL] > 0:
        overall_health = "critical"
        health_emoji = "🚨"
    elif alerts_count[AlertSeverity.WARNING] > 0:
        overall_health = "warning"
        health_emoji = "⚠️"
    else:
        overall_health = "healthy"
        health_emoji = "✅"
    
    print(f"{health_emoji} システム状態: {overall_health.upper()}")
    print(f"📊 分析メトリクス: {len(sample_metrics)}個")
    print(f"🚨 Critical: {alerts_count[AlertSeverity.CRITICAL]}件")
    print(f"⚠️  Warning: {alerts_count[AlertSeverity.WARNING]}件")
    print(f"ℹ️  Info: {alerts_count[AlertSeverity.INFO]}件")
    
    # 9. パフォーマンス情報
    print(f"\n⚡ 9. パフォーマンス情報")
    print("-" * 30)
    
    print(f"✅ 閾値チェック: {len(sample_metrics)}メトリクス処理")
    print(f"✅ トレンド分析: {len(increasing_values)}データポイント")
    print(f"✅ 異常検知: {len(anomaly_values)}データポイント")
    print(f"✅ 推奨生成: {len(recommendations)}項目")
    
    # 監査ログ記録
    from aals.core.logger import AuditAction, AuditLogEntry, audit_log
    
    audit_log(AuditLogEntry(
        action=AuditAction.VIEW,
        resource="prometheus_demo",
        result="success",
        details=f"Prometheus Analyzer デモ完了 - {overall_health}状態",
        risk_level="low"
    ))
    
    logger.info("Prometheus Analyzer demo completed",
                metrics_analyzed=len(sample_metrics),
                alerts_generated=len(all_alerts),
                system_health=overall_health)
    
    print(f"\n🎉 Module 4: Prometheus Analyzer デモ完了!")
    print(f"   📊 メトリクス分析: {len(sample_metrics)}個")
    print(f"   🚨 アラート検知: {len(all_alerts)}件")
    print(f"   💡 推奨事項: {len(recommendations)}個")
    print(f"   🎯 システム状態: {overall_health}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(demo_prometheus_analyzer())
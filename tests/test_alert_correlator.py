#!/usr/bin/env python3
"""
Tests for AALS Module 7: Alert Correlator
アラート相関分析・統合推奨生成モジュールのテスト
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from aals.modules.alert_correlator import (
    AlertCorrelator,
    AlertContext,
    CorrelatedAlert,
    IntegratedRecommendation,
    EscalationDecision,
    IncidentSeverity,
    EscalationLevel,
    CorrelationRule
)
from aals.modules.slack_alert_reader import SlackMessage
from aals.modules.prometheus_analyzer import AlertEvent, AlertSeverity
from aals.modules.github_issues_searcher import SimilarIssue, GitHubIssue, IssuePriority
from aals.modules.llm_wrapper import IncidentAnalysisResult


class TestAlertCorrelator:
    """Alert Correlator基本テスト"""
    
    def test_initialization(self, correlator):
        """初期化テスト"""
        assert correlator.correlation_window_minutes == 30
        assert correlator.confidence_threshold == 0.7
        assert correlator.max_correlations_per_window == 10
        assert isinstance(correlator.correlation_rules, list)
        assert isinstance(correlator.active_correlations, dict)
    
    def test_load_default_correlation_rules(self, correlator):
        """デフォルト相関ルール読み込みテスト"""
        assert len(correlator.correlation_rules) >= 3
        
        rule_names = [rule.name for rule in correlator.correlation_rules]
        assert "prometheus_slack_correlation" in rule_names
        assert "github_issue_pattern" in rule_names
        assert "multi_source_critical" in rule_names
        
        for rule in correlator.correlation_rules:
            assert isinstance(rule, CorrelationRule)
            assert rule.enabled
            assert 0 < rule.weight <= 1.0
            assert rule.time_window_minutes > 0
    
    @pytest.mark.asyncio
    async def test_verify_setup_success(self, correlator):
        """セットアップ確認成功テスト"""
        correlator.slack_reader.verify_setup = AsyncMock(return_value=True)
        correlator.prometheus_analyzer.verify_setup = AsyncMock(return_value=True)
        correlator.github_searcher.verify_setup = AsyncMock(return_value=True)
        correlator.llm_wrapper.verify_setup = AsyncMock(return_value=False)
        
        result = await correlator.verify_setup()
        assert result is True  # 3/4モジュール成功で OK
    
    @pytest.mark.asyncio
    async def test_verify_setup_failure(self, correlator):
        """セットアップ確認失敗テスト"""
        correlator.slack_reader.verify_setup = AsyncMock(return_value=False)
        correlator.prometheus_analyzer.verify_setup = AsyncMock(return_value=False)
        correlator.github_searcher.verify_setup = AsyncMock(return_value=True)
        correlator.llm_wrapper.verify_setup = AsyncMock(return_value=False)
        
        result = await correlator.verify_setup()
        assert result is False  # 1/4モジュールのみ成功


class TestAlertContextCollection:
    """アラートコンテキスト収集テスト"""
    
    @pytest.mark.asyncio
    async def test_collect_alert_contexts(self, correlator, sample_slack_messages, sample_prometheus_alerts):
        """アラートコンテキスト収集テスト"""
        # モックセットアップ
        correlator.slack_reader.get_recent_alerts = AsyncMock(return_value=sample_slack_messages)
        
        mock_system_report = Mock()
        mock_analysis = Mock()
        mock_analysis.alerts = sample_prometheus_alerts
        mock_analysis.trend = "increasing"
        mock_analysis.anomaly_score = 0.7
        mock_system_report.metric_analyses = [mock_analysis]
        
        correlator.prometheus_analyzer.analyze_system_health = AsyncMock(return_value=mock_system_report)
        
        # 実行
        contexts = await correlator.collect_alert_contexts(time_window_minutes=30)
        
        # 検証
        assert len(contexts) >= 2  # Slack + Prometheus
        
        slack_contexts = [c for c in contexts if c.source == "slack"]
        prometheus_contexts = [c for c in contexts if c.source == "prometheus"]
        
        assert len(slack_contexts) >= 1
        assert len(prometheus_contexts) >= 1
        
        # Slackコンテキスト検証
        slack_ctx = slack_contexts[0]
        assert slack_ctx.source == "slack"
        assert slack_ctx.severity == "critical"
        assert "API response time" in slack_ctx.content["message"]
        assert slack_ctx.confidence == 0.8
        
        # Prometheusコンテキスト検証
        prom_ctx = prometheus_contexts[0]
        assert prom_ctx.source == "prometheus"
        assert prom_ctx.severity == "warning"
        assert prom_ctx.content["metric_name"] == "cpu_usage_percent"
        assert prom_ctx.confidence == 0.9
    
    def test_is_within_time_window(self, correlator):
        """時間窓判定テスト"""
        now = datetime.now()
        
        # 窓内
        recent_time = now - timedelta(minutes=10)
        assert correlator._is_within_time_window(recent_time, 30) is True
        
        # 窓外
        old_time = now - timedelta(minutes=45)
        assert correlator._is_within_time_window(old_time, 30) is False
        
        # 境界値（30分ちょうどは窓外とする）
        boundary_time = now - timedelta(minutes=30)
        assert correlator._is_within_time_window(boundary_time, 30) is False
        
        # 境界値以内（29分）
        within_boundary = now - timedelta(minutes=29)
        assert correlator._is_within_time_window(within_boundary, 30) is True


class TestCorrelationAnalysis:
    """相関分析テスト"""
    
    
    def test_analyze_correlations(self, correlator, sample_contexts):
        """相関分析テスト"""
        correlations = correlator.analyze_correlations(sample_contexts)
        
        assert isinstance(correlations, list)
        assert len(correlations) >= 1
        
        correlation = correlations[0]
        assert isinstance(correlation, CorrelatedAlert)
        assert correlation.confidence_score >= 0.7  # 閾値以上
        assert correlation.severity in IncidentSeverity
        assert len(correlation.alert_contexts) >= 2
    
    def test_context_matches_condition(self, correlator):
        """コンテキスト条件マッチングテスト"""
        context = AlertContext(
            source="prometheus",
            timestamp=datetime.now(),
            severity="critical",
            content={"metric_name": "cpu_usage_percent"},
            confidence=0.9
        )
        
        # ソースマッチ
        condition1 = {"source": "prometheus"}
        assert correlator._context_matches_condition(context, condition1) is True
        
        # 重要度マッチ
        condition2 = {"source": "prometheus", "severity": ["critical", "warning"]}
        assert correlator._context_matches_condition(context, condition2) is True
        
        # マッチしない条件
        condition3 = {"source": "slack"}
        assert correlator._context_matches_condition(context, condition3) is False
        
        # 信頼度条件
        condition4 = {"source": "prometheus", "confidence": {"min": 0.8}}
        assert correlator._context_matches_condition(context, condition4) is True
        
        condition5 = {"source": "prometheus", "confidence": {"min": 0.95}}
        assert correlator._context_matches_condition(context, condition5) is False
    
    def test_severity_conversion(self, correlator):
        """重要度変換テスト"""
        # 重要度から数値
        assert correlator._severity_to_numeric("critical") == 5
        assert correlator._severity_to_numeric("warning") == 3
        assert correlator._severity_to_numeric("info") == 1
        
        # 数値から重要度
        assert correlator._numeric_to_incident_severity(5) == IncidentSeverity.CRITICAL
        assert correlator._numeric_to_incident_severity(3) == IncidentSeverity.MEDIUM
        assert correlator._numeric_to_incident_severity(1) == IncidentSeverity.INFO
    
    def test_deduplicate_correlations(self, correlator):
        """相関重複排除テスト"""
        now = datetime.now()
        
        correlation1 = CorrelatedAlert(
            correlation_id="corr1",
            primary_source="slack",
            related_sources=["prometheus"],
            severity=IncidentSeverity.CRITICAL,
            confidence_score=0.8,
            time_window=(now - timedelta(minutes=10), now - timedelta(minutes=5)),
            alert_contexts=[],
            correlation_evidence={},
            estimated_impact={},
            timestamp=now - timedelta(minutes=5)
        )
        
        correlation2 = CorrelatedAlert(
            correlation_id="corr2", 
            primary_source="slack",
            related_sources=["prometheus"],
            severity=IncidentSeverity.CRITICAL,
            confidence_score=0.9,  # より高い信頼度
            time_window=(now - timedelta(minutes=8), now - timedelta(minutes=3)),
            alert_contexts=[],
            correlation_evidence={},
            estimated_impact={},
            timestamp=now - timedelta(minutes=3)  # 近い時間
        )
        
        correlations = [correlation1, correlation2]
        unique = correlator._deduplicate_correlations(correlations)
        
        assert len(unique) == 1
        assert unique[0].correlation_id == "corr2"  # より高い信頼度が保持される


class TestRecommendationGeneration:
    """推奨アクション生成テスト"""
    
    
    
    
    @pytest.mark.asyncio
    async def test_generate_integrated_recommendations(
        self, 
        correlator, 
        sample_correlation,
        sample_llm_analysis,
        sample_similar_cases
    ):
        """統合推奨生成テスト"""
        # モックセットアップ
        correlator._get_llm_analysis = AsyncMock(return_value=sample_llm_analysis)
        correlator._find_similar_cases = AsyncMock(return_value=sample_similar_cases)
        
        # 実行
        recommendations = await correlator.generate_integrated_recommendations(sample_correlation)
        
        # 検証
        assert isinstance(recommendations, IntegratedRecommendation)
        assert recommendations.correlation_id == sample_correlation.correlation_id
        assert recommendations.priority >= 1
        assert len(recommendations.immediate_actions) >= 3
        assert len(recommendations.investigation_steps) >= 3
        assert len(recommendations.mitigation_strategies) >= 3
        assert len(recommendations.preventive_measures) >= 3
        assert recommendations.estimated_resolution_time > 0
        assert len(recommendations.required_skills) >= 1
        assert isinstance(recommendations.risk_assessment, dict)
    
    def test_estimate_resolution_time(self, correlator, sample_correlation, sample_similar_cases):
        """解決時間推定テスト"""
        # 類似ケースなし
        time_no_cases = correlator._estimate_resolution_time(sample_correlation, [])
        assert time_no_cases == 60  # CRITICAL基本時間
        
        # 類似ケースあり
        time_with_cases = correlator._estimate_resolution_time(sample_correlation, sample_similar_cases)
        assert time_with_cases != time_no_cases  # 調整されている
        assert time_with_cases > 0
    
    def test_identify_required_skills(self, correlator, sample_correlation):
        """必要スキル特定テスト"""
        skills = correlator._identify_required_skills(sample_correlation)
        
        assert isinstance(skills, list)
        assert len(skills) >= 1
        assert "Incident Communication" in skills
    
    def test_assess_risks(self, correlator, sample_correlation):
        """リスク評価テスト"""
        risks = correlator._assess_risks(sample_correlation)
        
        assert isinstance(risks, dict)
        assert "data_loss_risk" in risks
        assert "service_degradation" in risks
        assert "user_impact" in risks
        assert "escalation_probability" in risks
        assert "automated_recovery_possible" in risks
        
        # CRITICALレベルでの適切な評価
        assert risks["automated_recovery_possible"] is False  # CRITICAL は自動復旧困難
    
    def test_calculate_priority(self, correlator, sample_correlation):
        """優先度計算テスト"""
        priority = correlator._calculate_priority(sample_correlation)
        
        assert isinstance(priority, int)
        assert 1 <= priority <= 5
        assert priority == 1  # CRITICAL + 高信頼度 = 最高優先度


class TestEscalationDecision:
    """エスカレーション判定テスト"""
    
    
    
    def test_escalation_decision_critical(self, correlator, critical_correlation):
        """クリティカルエスカレーション判定テスト"""
        decision = correlator.make_escalation_decision(critical_correlation)
        
        assert isinstance(decision, EscalationDecision)
        assert decision.correlation_id == critical_correlation.correlation_id
        assert decision.escalation_level == EscalationLevel.IMMEDIATE_ACTION
        assert decision.human_intervention_required is True
        assert decision.automated_actions_allowed is False
        assert decision.time_limit_minutes == 15
        assert len(decision.required_approvals) >= 1
        assert len(decision.notification_targets) >= 1
    
    def test_escalation_decision_medium(self, correlator, medium_correlation):
        """中程度エスカレーション判定テスト"""
        decision = correlator.make_escalation_decision(medium_correlation)
        
        assert isinstance(decision, EscalationDecision)
        assert decision.escalation_level == EscalationLevel.MONITOR_ONLY
        assert decision.human_intervention_required is False
        assert decision.automated_actions_allowed is True
        assert decision.time_limit_minutes is None
    
    def test_escalation_reasoning(self, correlator, critical_correlation):
        """エスカレーション理由テスト"""
        decision = correlator.make_escalation_decision(critical_correlation)
        
        assert "critical" in decision.reasoning.lower()
        assert "0.90" in decision.reasoning  # 信頼度
        assert "3" in decision.reasoning  # ソース数（1 primary + 2 related）


class TestWorkflowIntegration:
    """ワークフロー統合テスト"""
    
    @pytest.mark.asyncio
    async def test_process_incident_workflow(self, correlator):
        """インシデントワークフロー処理テスト"""
        # モックセットアップ
        sample_contexts = [
            AlertContext(
                source="slack",
                timestamp=datetime.now() - timedelta(minutes=5),
                severity="critical",
                content={"message": "Test alert"},
                confidence=0.8
            )
        ]
        
        correlator.collect_alert_contexts = AsyncMock(return_value=sample_contexts)
        correlator.analyze_correlations = Mock(return_value=[])
        
        # 実行
        result = await correlator.process_incident_workflow(time_window_minutes=30)
        
        # 検証
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "contexts_analyzed" in result
        assert "correlations_found" in result
        assert "workflow_results" in result
        assert result["contexts_analyzed"] == 1
        assert result["correlations_found"] == 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_correlation(self, correlator):
        """相関ありワークフローテスト"""
        # サンプルデータ作成
        sample_contexts = [
            AlertContext(
                source="slack",
                timestamp=datetime.now() - timedelta(minutes=5),
                severity="critical",
                content={"message": "API critical", "keywords": ["api", "critical"]},
                confidence=0.9
            )
        ]
        
        sample_correlation = CorrelatedAlert(
            correlation_id="test_123",
            primary_source="slack",
            related_sources=[],
            severity=IncidentSeverity.CRITICAL,
            confidence_score=0.85,
            time_window=(datetime.now() - timedelta(minutes=10), datetime.now()),
            alert_contexts=sample_contexts,
            correlation_evidence={},
            estimated_impact={}
        )
        
        # モックセットアップ
        correlator.collect_alert_contexts = AsyncMock(return_value=sample_contexts)
        correlator.analyze_correlations = Mock(return_value=[sample_correlation])
        correlator.generate_integrated_recommendations = AsyncMock(return_value=Mock(to_dict=Mock(return_value={})))
        correlator.make_escalation_decision = Mock(return_value=Mock(to_dict=Mock(return_value={})))
        
        # 実行
        result = await correlator.process_incident_workflow()
        
        # 検証
        assert result["correlations_found"] == 1
        assert len(result["workflow_results"]) == 1
        assert "test_123" in correlator.active_correlations


class TestDataStructures:
    """データ構造テスト"""
    
    def test_alert_context_to_dict(self):
        """AlertContextシリアライゼーションテスト"""
        context = AlertContext(
            source="test",
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            severity="critical",
            content={"key": "value"},
            confidence=0.8
        )
        
        data = context.to_dict()
        assert isinstance(data, dict)
        assert data["source"] == "test"
        assert data["severity"] == "critical"
        assert data["confidence"] == 0.8
        assert "2023-01-01T12:00:00" in data["timestamp"]
    
    def test_correlated_alert_to_dict(self):
        """CorrelatedAlertシリアライゼーションテスト"""
        alert = CorrelatedAlert(
            correlation_id="test_123",
            primary_source="slack",
            related_sources=["prometheus"],
            severity=IncidentSeverity.CRITICAL,
            confidence_score=0.85,
            time_window=(datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 1, 12, 10)),
            alert_contexts=[],
            correlation_evidence={"test": "evidence"},
            estimated_impact={"impact": "high"}
        )
        
        data = alert.to_dict()
        assert isinstance(data, dict)
        assert data["correlation_id"] == "test_123"
        assert data["severity"] == "critical"
        assert data["confidence_score"] == 0.85
        assert len(data["time_window"]) == 2
    
    def test_integrated_recommendation_to_dict(self):
        """IntegratedRecommendationシリアライゼーションテスト"""
        recommendation = IntegratedRecommendation(
            correlation_id="test_123",
            priority=1,
            immediate_actions=["action1", "action2"],
            investigation_steps=["step1"],
            mitigation_strategies=["strategy1"],
            preventive_measures=["measure1"],
            estimated_resolution_time=60,
            required_skills=["skill1"],
            risk_assessment={"risk": "medium"},
            automation_candidates=["auto1"],
            escalation_triggers=["trigger1"]
        )
        
        data = recommendation.to_dict()
        assert isinstance(data, dict)
        assert data["correlation_id"] == "test_123"
        assert data["priority"] == 1
        assert len(data["immediate_actions"]) == 2
        assert data["estimated_resolution_time"] == 60
    
    def test_escalation_decision_to_dict(self):
        """EscalationDecisionシリアライゼーションテスト"""
        decision = EscalationDecision(
            correlation_id="test_123",
            escalation_level=EscalationLevel.IMMEDIATE_ACTION,
            reasoning="Test reasoning",
            required_approvals=["approval1"],
            time_limit_minutes=15,
            notification_targets=["target1"],
            automated_actions_allowed=False,
            human_intervention_required=True
        )
        
        data = decision.to_dict()
        assert isinstance(data, dict)
        assert data["correlation_id"] == "test_123"
        assert data["escalation_level"] == "immediate_action"
        assert data["reasoning"] == "Test reasoning"
        assert data["time_limit_minutes"] == 15
        assert data["automated_actions_allowed"] is False
        assert data["human_intervention_required"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
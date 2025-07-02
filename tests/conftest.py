#!/usr/bin/env python3
"""
Common test fixtures for AALS tests
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from aals.modules.alert_correlator import (
    AlertCorrelator, AlertContext, CorrelatedAlert, IncidentSeverity
)
from aals.integrations.slack_client import SlackMessage
from aals.modules.prometheus_analyzer import AlertEvent, AlertSeverity
from aals.modules.github_issues_searcher import SimilarIssue, GitHubIssue, IssuePriority
from aals.modules.llm_wrapper import IncidentAnalysisResult


@pytest.fixture
def mock_config():
    """Mock configuration for Alert Correlator"""
    config = Mock()
    config.config = {
        "correlation_window_minutes": 30,
        "confidence_threshold": 0.7,
        "max_correlations_per_window": 10,
        "correlation_rules": []
    }
    return config


@pytest.fixture
def mock_modules():
    """Mock dependent modules"""
    return {
        "slack_reader": Mock(),
        "prometheus_analyzer": Mock(), 
        "github_searcher": Mock(),
        "llm_wrapper": Mock()
    }


@pytest.fixture
def correlator(mock_config, mock_modules):
    """Alert Correlator instance with mocked dependencies"""
    with patch('aals.modules.alert_correlator.get_config_manager') as mock_config_manager:
        mock_config_manager.return_value.get_module_config.return_value = mock_config
        
        with patch.multiple(
            'aals.modules.alert_correlator',
            SlackAlertReader=Mock(return_value=mock_modules["slack_reader"]),
            PrometheusAnalyzer=Mock(return_value=mock_modules["prometheus_analyzer"]),
            GitHubIssuesSearcher=Mock(return_value=mock_modules["github_searcher"]),
            LLMWrapper=Mock(return_value=mock_modules["llm_wrapper"])
        ):
            correlator = AlertCorrelator()
            correlator.slack_reader = mock_modules["slack_reader"]
            correlator.prometheus_analyzer = mock_modules["prometheus_analyzer"]
            correlator.github_searcher = mock_modules["github_searcher"]
            correlator.llm_wrapper = mock_modules["llm_wrapper"]
            return correlator


@pytest.fixture
def sample_slack_messages():
    """Sample Slack messages for testing"""
    msg1 = SlackMessage(
        channel="C123",
        channel_name="#alerts",
        timestamp="1234567890.123",
        text="CRITICAL: API response time exceeding 5 seconds",
        user="monitoring-bot",
        thread_ts=None,
        reactions=[]
    )
    msg1.is_alert = True
    msg1.alert_level = "critical"
    
    msg2 = SlackMessage(
        channel="C123", 
        channel_name="#alerts",
        timestamp="1234567891.123",
        text="INFO: Deployment completed successfully",
        user="deploy-bot",
        thread_ts=None,
        reactions=[]
    )
    msg2.is_alert = False
    msg2.alert_level = "info"
    
    return [msg1, msg2]


@pytest.fixture
def sample_prometheus_alerts():
    """Sample Prometheus alerts for testing"""
    return [
        AlertEvent(
            metric_name="cpu_usage_percent",
            current_value=85.5,
            threshold_value=80.0,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now() - timedelta(minutes=3),
            labels={"instance": "web-01"},
            message="CPU usage is above warning threshold",
            comparison=">"
        )
    ]


@pytest.fixture
def sample_github_issue():
    """Sample GitHub issue for testing"""
    return GitHubIssue(
        number=123,
        title="API performance degradation", 
        body="Similar API issue resolved",
        state="closed",
        labels=["bug", "performance"],
        assignees=["dev-team"],
        created_at=datetime.now() - timedelta(days=5),
        updated_at=datetime.now() - timedelta(days=3),
        closed_at=datetime.now() - timedelta(days=2),
        author="system",
        comments_count=5,
        url="https://github.com/repo/issues/123",
        repository="company/api",
        priority=IssuePriority.HIGH,
        resolution_time=timedelta(hours=8)
    )


@pytest.fixture
def sample_similar_cases(sample_github_issue):
    """Sample similar cases for testing"""
    return [SimilarIssue(
        issue=sample_github_issue,
        similarity_score=0.8,
        matching_keywords=["api", "performance"],
        relevance_reason="Similar API performance issue"
    )]


@pytest.fixture
def sample_contexts():
    """Sample alert contexts for testing"""
    now = datetime.now()
    return [
        AlertContext(
            source="slack",
            timestamp=now - timedelta(minutes=5),
            severity="critical",
            content={
                "message": "API response time critical",
                "channel": "#alerts",
                "keywords": ["api", "critical", "response"]
            },
            confidence=0.8
        ),
        AlertContext(
            source="prometheus",
            timestamp=now - timedelta(minutes=3),
            severity="warning",
            content={
                "metric_name": "http_response_time_seconds",
                "current_value": 6.2,
                "threshold_value": 5.0,
                "trend": "increasing"
            },
            confidence=0.9
        ),
        AlertContext(
            source="prometheus",
            timestamp=now - timedelta(minutes=2),
            severity="critical",
            content={
                "metric_name": "cpu_usage_percent",
                "current_value": 95.0,
                "threshold_value": 90.0
            },
            confidence=0.9
        )
    ]


@pytest.fixture
def sample_correlation():
    """Sample correlation for testing"""
    now = datetime.now()
    contexts = [
        AlertContext(
            source="slack",
            timestamp=now - timedelta(minutes=5),
            severity="critical",
            content={"message": "API down", "keywords": ["api", "critical"]},
            confidence=0.9
        )
    ]
    
    return CorrelatedAlert(
        correlation_id="test_corr_123",
        primary_source="slack",
        related_sources=["prometheus"],
        severity=IncidentSeverity.CRITICAL,
        confidence_score=0.85,
        time_window=(now - timedelta(minutes=10), now),
        alert_contexts=contexts,
        correlation_evidence={"rule_name": "test_rule"},
        estimated_impact={"user_impact_level": "high"}
    )


@pytest.fixture
def sample_llm_analysis():
    """Sample LLM analysis for testing"""
    return IncidentAnalysisResult(
        root_cause="High API response times due to database overload",
        mitigation_steps=[
            "Scale database connections",
            "Implement connection pooling",
            "Add caching layer"
        ],
        prevention_strategies=[
            "Monitor database performance",
            "Implement auto-scaling",
            "Regular performance testing"
        ],
        related_patterns=["database_overload", "api_performance"],
        severity_assessment="CRITICAL",
        estimated_impact="200+ users affected",
        confidence_score=0.85
    )


@pytest.fixture
def critical_correlation():
    """Critical correlation for testing"""
    return CorrelatedAlert(
        correlation_id="critical_123",
        primary_source="slack",
        related_sources=["prometheus", "github"],
        severity=IncidentSeverity.CRITICAL,
        confidence_score=0.9,
        time_window=(datetime.now() - timedelta(minutes=10), datetime.now()),
        alert_contexts=[],
        correlation_evidence={},
        estimated_impact={}
    )


@pytest.fixture
def medium_correlation():
    """Medium correlation for testing"""
    return CorrelatedAlert(
        correlation_id="medium_123",
        primary_source="prometheus",
        related_sources=[],
        severity=IncidentSeverity.MEDIUM,
        confidence_score=0.8,
        time_window=(datetime.now() - timedelta(minutes=5), datetime.now()),
        alert_contexts=[],
        correlation_evidence={},
        estimated_impact={}
    )
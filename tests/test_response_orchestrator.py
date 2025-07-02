#!/usr/bin/env python3
"""
Tests for AALS Module 9: Response Orchestrator
全モジュール統合制御・ワークフロー管理モジュールのテスト
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from aals.modules.response_orchestrator import (
    ResponseOrchestrator,
    IncidentEvent,
    WorkflowInstance,
    WorkflowStep,
    IncidentResponse,
    WorkflowStage,
    WorkflowStatus,
    AutomationLevel,
    WorkflowTemplate
)


# Module-level fixtures available to all test classes
@pytest.fixture
def mock_config():
    """モック設定"""
    config = Mock()
    config.enabled = True
    config.config = {
        "workflow": {
            "execution": {
                "max_concurrent_workflows": 10,
                "max_steps_per_workflow": 50,
                "default_step_timeout_minutes": 30,
                "workflow_timeout_hours": 24
            },
            "automation_levels": {
                "manual": {"auto_execute": False, "require_approval": True},
                "semi_auto": {"auto_execute": False, "require_approval": True},
                "auto_monitor": {"auto_execute": True, "require_approval": False},
                "full_auto": {"auto_execute": True, "require_approval": False}
            }
        },
        "approval": {
            "approval_timeout_minutes": 60,
            "approval_hierarchy": {
                "level_1": {"approvers": ["sre-team"]},
                "level_2": {"approvers": ["sre-team", "team-lead"]},
                "level_3": {"approvers": ["sre-team", "team-lead", "on-call-engineer"]}
            }
        },
        "execution": {
            "concurrency": {"max_parallel_workflows": 5},
            "retry": {"enabled": True, "max_attempts": 3},
            "timeouts": {"step_timeout_minutes": 30}
        },
        "templates": {}
    }
    return config


@pytest.fixture
def mock_modules():
    """依存モジュールのモック"""
    # AsyncMockで非同期メソッドをモック
    slack_reader = Mock()
    slack_reader.get_recent_alerts = AsyncMock(return_value=[])
    slack_reader.verify_setup = AsyncMock(return_value=True)
    
    prometheus_analyzer = Mock()
    prometheus_analyzer.analyze_system_health = AsyncMock(return_value=Mock())
    prometheus_analyzer.analyze_metric_trend = AsyncMock(return_value=Mock())
    prometheus_analyzer.verify_setup = AsyncMock(return_value=True)
    
    github_searcher = Mock()
    github_searcher.search_similar_incidents = AsyncMock(return_value=[])
    github_searcher.verify_setup = AsyncMock(return_value=True)
    
    llm_wrapper = Mock()
    llm_wrapper.analyze_incident = AsyncMock(return_value=Mock())
    llm_wrapper.analyze_metrics = AsyncMock(return_value=Mock())
    llm_wrapper.verify_setup = AsyncMock(return_value=True)
    
    alert_correlator = Mock()
    alert_correlator.collect_alert_contexts = AsyncMock(return_value=[])
    alert_correlator.verify_setup = AsyncMock(return_value=True)
    
    ssh_executor = Mock()
    ssh_executor.execute_command = AsyncMock(return_value=Mock())
    ssh_executor.verify_setup = AsyncMock(return_value=True)
    
    return {
        "slack_reader": slack_reader,
        "prometheus_analyzer": prometheus_analyzer,
        "github_searcher": github_searcher,
        "llm_wrapper": llm_wrapper,
        "alert_correlator": alert_correlator,
        "ssh_executor": ssh_executor
    }


@pytest.fixture
def orchestrator(mock_config, mock_modules):
    """Response Orchestratorインスタンス"""
    with patch('aals.modules.response_orchestrator.get_config_manager') as mock_config_manager:
        mock_config_manager.return_value.get_module_config.return_value = mock_config
        
        with patch.multiple(
            'aals.modules.response_orchestrator',
            SlackAlertReader=Mock(return_value=mock_modules["slack_reader"]),
            PrometheusAnalyzer=Mock(return_value=mock_modules["prometheus_analyzer"]),
            GitHubIssuesSearcher=Mock(return_value=mock_modules["github_searcher"]),
            LLMWrapper=Mock(return_value=mock_modules["llm_wrapper"]),
            AlertCorrelator=Mock(return_value=mock_modules["alert_correlator"]),
            SSHExecutor=Mock(return_value=mock_modules["ssh_executor"])
        ):
            orchestrator = ResponseOrchestrator()
            # 直接モジュールを設定
            orchestrator.slack_reader = mock_modules["slack_reader"]
            orchestrator.prometheus_analyzer = mock_modules["prometheus_analyzer"]
            orchestrator.github_searcher = mock_modules["github_searcher"]
            orchestrator.llm_wrapper = mock_modules["llm_wrapper"]
            orchestrator.alert_correlator = mock_modules["alert_correlator"]
            orchestrator.ssh_executor = mock_modules["ssh_executor"]
            return orchestrator


@pytest.fixture
def sample_incident_event():
    """サンプルインシデントイベント"""
    return IncidentEvent(
        source="slack",
        event_type="alert",
        severity="critical",
        title="API Response Time Critical",
        description="API response times are exceeding 5 seconds",
        metadata={"channel": "#alerts", "user": "monitoring-bot"},
        correlation_id="corr_123"
    )


class TestResponseOrchestrator:
    """Response Orchestrator基本テスト"""
    
    def test_initialization(self, orchestrator):
        """初期化テスト"""
        assert orchestrator.max_concurrent_workflows == 10
        assert orchestrator.workflow_timeout_minutes == 60
        assert isinstance(orchestrator.active_workflows, dict)
        assert isinstance(orchestrator.workflow_history, list)
        assert isinstance(orchestrator.modules, dict)
        assert len(orchestrator.modules) == 6
    
    def test_workflow_templates(self):
        """ワークフローテンプレートテスト"""
        # 標準インシデント対応ワークフロー
        standard_steps = WorkflowTemplate.create_standard_incident_workflow()
        assert len(standard_steps) >= 6
        assert all(isinstance(step, WorkflowStep) for step in standard_steps)
        
        # パフォーマンス調査ワークフロー
        performance_steps = WorkflowTemplate.create_performance_investigation_workflow()
        assert len(performance_steps) >= 3
        assert all(isinstance(step, WorkflowStep) for step in performance_steps)
    
    @pytest.mark.asyncio
    async def test_verify_setup_success(self, orchestrator):
        """セットアップ確認成功テスト"""
        # 全モジュールが成功
        for module in [orchestrator.slack_reader, orchestrator.prometheus_analyzer, 
                      orchestrator.github_searcher, orchestrator.llm_wrapper,
                      orchestrator.alert_correlator, orchestrator.ssh_executor]:
            module.verify_setup = AsyncMock(return_value=True)
        
        result = await orchestrator.verify_setup()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_setup_partial_failure(self, orchestrator):
        """セットアップ確認部分失敗テスト"""
        # 半分のモジュールが失敗
        orchestrator.slack_reader.verify_setup = AsyncMock(return_value=True)
        orchestrator.prometheus_analyzer.verify_setup = AsyncMock(return_value=True)
        orchestrator.github_searcher.verify_setup = AsyncMock(return_value=True)
        orchestrator.llm_wrapper.verify_setup = AsyncMock(return_value=False)
        orchestrator.alert_correlator.verify_setup = AsyncMock(return_value=False)
        orchestrator.ssh_executor.verify_setup = AsyncMock(return_value=False)
        
        result = await orchestrator.verify_setup()
        assert result is True  # 3/6成功で OK


class TestWorkflowManagement:
    """ワークフロー管理テスト"""
    
    @pytest.fixture
    def sample_workflow_request(self):
        """サンプルワークフローリクエスト"""
        return {
            "workflow_type": "incident_response",
            "automation_level": AutomationLevel.SEMI_AUTO,
            "priority": 1,
            "requested_by": "sre-engineer",
            "context": {"incident_severity": "critical"}
        }
    
    @pytest.mark.asyncio
    async def test_handle_incident(self, orchestrator, sample_incident_event):
        """インシデント処理テスト"""
        instance = await orchestrator.handle_incident(
            incident_event=sample_incident_event,
            automation_level=AutomationLevel.SEMI_AUTO
        )
        
        assert isinstance(instance, WorkflowInstance)
        assert instance.automation_level == AutomationLevel.SEMI_AUTO
        assert instance.status == WorkflowStatus.PENDING
        assert instance.incident_event == sample_incident_event
        assert len(instance.steps) >= 6  # incident_responseテンプレートのステップ数
    
    @pytest.mark.asyncio
    async def test_workflow_steps_creation(self, orchestrator, sample_incident_event):
        """ワークフローステップ作成テスト"""
        instance = await orchestrator.handle_incident(
            incident_event=sample_incident_event,
            automation_level=AutomationLevel.SEMI_AUTO
        )
        
        step_stages = [step.stage for step in instance.steps]
        assert WorkflowStage.DETECTION in step_stages
        assert WorkflowStage.CORRELATION in step_stages
        assert WorkflowStage.ANALYSIS in step_stages
        assert WorkflowStage.PLANNING in step_stages
        assert WorkflowStage.EXECUTION in step_stages
        assert WorkflowStage.MONITORING in step_stages
        
        # 各ステップの基本属性確認
        for step in instance.steps:
            assert isinstance(step, WorkflowStep)
            assert step.step_id is not None
            assert step.stage is not None
            assert step.name != ""
    
    @pytest.mark.asyncio
    async def test_handle_incident_starts_workflow(self, orchestrator, sample_incident_event):
        """インシデント処理でワークフローが開始されることのテスト"""
        instance = await orchestrator.handle_incident(
            incident_event=sample_incident_event,
            automation_level=AutomationLevel.SEMI_AUTO
        )
        
        # ワークフローが作成され、アクティブワークフローに追加される
        assert instance.workflow_id in orchestrator.active_workflows
        assert instance.started_at is not None
    
    @pytest.mark.asyncio
    async def test_get_workflow_status(self, orchestrator, sample_incident_event):
        """ワークフローステータス取得テスト"""
        instance = await orchestrator.handle_incident(
            incident_event=sample_incident_event,
            automation_level=AutomationLevel.SEMI_AUTO
        )
        
        status = orchestrator.get_workflow_status(instance.workflow_id)
        assert isinstance(status, WorkflowInstance)
        assert status.workflow_id == instance.workflow_id
        assert status.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]
        assert len(status.steps) >= 6
    
    def test_get_workflow_status_not_found(self, orchestrator):
        """存在しないワークフローステータス取得テスト"""
        status = orchestrator.get_workflow_status("nonexistent")
        assert status is None


class TestStepExecution:
    """ステップ実行テスト"""
    
    @pytest.fixture
    def sample_detection_step(self):
        """サンプル検知ステップ"""
        return WorkflowStep(
            stage=WorkflowStage.DETECTION,
            name="Alert Detection",
            description="Detect and collect alerts from all sources",
            module="alert_correlator",
            action="collect_alert_contexts",
            parameters={"time_window_minutes": 30}
        )
    
    @pytest.fixture
    def sample_execution_step(self):
        """サンプル実行ステップ"""
        return WorkflowStep(
            stage=WorkflowStage.EXECUTION,
            name="Execute Commands",
            description="Execute remediation commands",
            module="ssh_executor",
            action="execute_command",
            parameters={"dry_run": True}
        )
    
    def test_workflow_step_creation(self, sample_detection_step):
        """ワークフローステップ作成テスト"""
        assert sample_detection_step.stage == WorkflowStage.DETECTION
        assert sample_detection_step.name == "Alert Detection"
        assert sample_detection_step.module == "alert_correlator"
        assert sample_detection_step.action == "collect_alert_contexts"
        assert sample_detection_step.status == "pending"
    
    def test_workflow_step_serialization(self, sample_execution_step):
        """ワークフローステップシリアライゼーションテスト"""
        data = sample_execution_step.to_dict()
        assert isinstance(data, dict)
        assert data["stage"] == "execution"
        assert data["name"] == "Execute Commands"
        assert data["module"] == "ssh_executor"
        assert data["action"] == "execute_command"


class TestApprovalManagement:
    """承認管理テスト"""
    
    @pytest.mark.asyncio
    async def test_approve_workflow_step(self, orchestrator, sample_incident_event):
        """ワークフローステップ承認テスト"""
        # インシデント処理開始 (承認が必要なステップを含む)
        instance = await orchestrator.handle_incident(
            incident_event=sample_incident_event,
            automation_level=AutomationLevel.SEMI_AUTO
        )
        
        # 承認が必要なワークフローを設定
        if instance.approvals:
            approval_id = list(instance.approvals.keys())[0]
            
            # 承認実行
            result = await orchestrator.approve_workflow_step(
                workflow_id=instance.workflow_id,
                approval_id=approval_id,
                approver="sre-team",
                decision=True
            )
            
            assert result is True
    
    @pytest.mark.asyncio 
    async def test_approve_workflow_step_not_found(self, orchestrator):
        """存在しないワークフロー承認テスト"""
        result = await orchestrator.approve_workflow_step(
            workflow_id="nonexistent",
            approval_id="nonexistent", 
            approver="sre-team",
            decision=True
        )
        assert result is False
    
    def test_get_pending_approvals(self, orchestrator):
        """承認待ちリスト取得テスト"""
        pending = orchestrator.get_pending_approvals()
        assert isinstance(pending, list)


class TestDashboardGeneration:
    """ダッシュボード生成テスト"""
    
    @pytest.mark.asyncio
    async def test_generate_dashboard_data(self, orchestrator, sample_incident_event):
        """ダッシュボードデータ生成テスト"""
        # サンプルワークフロー追加
        instance = await orchestrator.handle_incident(
            incident_event=sample_incident_event,
            automation_level=AutomationLevel.SEMI_AUTO
        )
        
        # ダッシュボードデータ生成
        dashboard = orchestrator.generate_dashboard_data()
        
        assert isinstance(dashboard, dict)
        assert "active_workflows" in dashboard
        assert "recent_history" in dashboard
        assert "performance" in dashboard
        assert "pending_approvals" in dashboard
        assert dashboard["active_workflows"]["total"] >= 1
    
    def test_dashboard_basic_structure(self, orchestrator):
        """ダッシュボード基本構造テスト"""
        dashboard = orchestrator.generate_dashboard_data()
        
        assert isinstance(dashboard, dict)
        assert "timestamp" in dashboard
        assert "active_workflows" in dashboard
        assert "recent_history" in dashboard  
        assert "performance" in dashboard
        assert "pending_approvals" in dashboard
        
        # アクティブワークフロー構造確認
        active = dashboard["active_workflows"]
        assert "total" in active
        assert "waiting_approval" in active
        assert "running" in active
        
        # 履歴構造確認
        history = dashboard["recent_history"]
        assert "completed" in history
        assert "failed" in history
        assert "success_rate" in history
    
    def test_dashboard_performance_metrics(self, orchestrator):
        """ダッシュボードパフォーマンスメトリクステスト"""
        dashboard = orchestrator.generate_dashboard_data()
        
        performance = dashboard["performance"]
        assert "avg_execution_time_seconds" in performance
        assert "total_workflows_processed" in performance
        assert isinstance(performance["avg_execution_time_seconds"], (int, float))
        assert isinstance(performance["total_workflows_processed"], int)


class TestIntegrationScenarios:
    """統合シナリオテスト"""
    
    @pytest.mark.asyncio
    async def test_full_incident_response_workflow(self, orchestrator):
        """完全インシデント対応ワークフローテスト"""
        # インシデントイベント作成
        incident_event = IncidentEvent(
            source="slack",
            event_type="alert",
            severity="critical",
            title="API Down",
            description="API is completely down",
            correlation_id="incident_123"
        )
        
        # 自動実行可能ステップのモック設定
        orchestrator.slack_reader.get_recent_alerts = AsyncMock(return_value=[])
        orchestrator.prometheus_analyzer.analyze_system_health = AsyncMock(return_value=Mock())
        orchestrator.alert_correlator.collect_alert_contexts = AsyncMock(return_value=[])
        orchestrator.llm_wrapper.analyze_incident = AsyncMock(return_value=Mock())
        orchestrator.github_searcher.search_similar_incidents = AsyncMock(return_value=[])
        
        # インシデント処理開始
        instance = await orchestrator.handle_incident(
            incident_event=incident_event,
            automation_level=AutomationLevel.SEMI_AUTO
        )
        
        # ワークフローが開始されたことを確認
        assert instance.workflow_id in orchestrator.active_workflows
        assert instance.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]
        
        # ステータス確認
        status = orchestrator.get_workflow_status(instance.workflow_id)
        assert isinstance(status, WorkflowInstance)
        assert status.workflow_id == instance.workflow_id
    
    @pytest.mark.asyncio
    async def test_performance_investigation_workflow(self, orchestrator):
        """パフォーマンス調査ワークフローテスト"""
        # パフォーマンス問題イベント
        performance_event = IncidentEvent(
            source="prometheus",
            event_type="metric_threshold",
            severity="medium",
            title="High CPU Usage",
            description="CPU usage consistently above 80%"
        )
        
        # モック設定
        orchestrator.prometheus_analyzer.analyze_system_health = AsyncMock(return_value=Mock())
        orchestrator.prometheus_analyzer.analyze_metric_trend = AsyncMock(return_value=Mock())
        orchestrator.llm_wrapper.analyze_metrics = AsyncMock(return_value=Mock())
        orchestrator.github_searcher.search_similar_incidents = AsyncMock(return_value=[])
        
        # パフォーマンス調査ワークフロー開始
        instance = await orchestrator.handle_incident(
            incident_event=performance_event,
            automation_level=AutomationLevel.AUTO_MONITOR,
            workflow_template="performance"
        )
        
        # ワークフローが開始されたことを確認
        assert instance.workflow_id in orchestrator.active_workflows
        assert instance.automation_level == AutomationLevel.AUTO_MONITOR
        
        # ステータス確認
        status = orchestrator.get_workflow_status(instance.workflow_id)
        assert isinstance(status, WorkflowInstance)
        assert status.workflow_id == instance.workflow_id
    
    @pytest.mark.asyncio
    async def test_workflow_cancel(self, orchestrator, sample_incident_event):
        """ワークフローキャンセルテスト"""
        # ワークフロー開始
        instance = await orchestrator.handle_incident(
            incident_event=sample_incident_event,
            automation_level=AutomationLevel.SEMI_AUTO
        )
        
        # キャンセル実行
        result = await orchestrator.cancel_workflow(
            workflow_id=instance.workflow_id,
            reason="Test cancellation"
        )
        
        assert result is True
        # キャンセル後はアクティブワークフローから削除される
        assert instance.workflow_id not in orchestrator.active_workflows


class TestDataStructures:
    """データ構造テスト"""
    
    def test_incident_event_to_dict(self):
        """IncidentEventシリアライゼーションテスト"""
        event = IncidentEvent(
            source="test",
            event_type="alert",
            severity="critical",
            title="Test Event",
            description="Test description",
            metadata={"key": "value"},
            correlation_id="corr_123"
        )
        
        data = event.to_dict()
        assert isinstance(data, dict)
        assert data["source"] == "test"
        assert data["severity"] == "critical"
        assert data["title"] == "Test Event"
        assert data["correlation_id"] == "corr_123"
        assert isinstance(data["timestamp"], str)
    
    def test_workflow_instance_to_dict(self):
        """WorkflowInstanceシリアライゼーションテスト"""
        event = IncidentEvent(source="test", event_type="alert", severity="info", title="Test")
        
        instance = WorkflowInstance(
            incident_event=event,
            automation_level=AutomationLevel.SEMI_AUTO,
            status=WorkflowStatus.RUNNING
        )
        
        data = instance.to_dict()
        assert isinstance(data, dict)
        assert data["automation_level"] == "semi_auto"
        assert data["status"] == "running"
        assert "steps" in data
        assert "workflow_id" in data
        assert "incident_event" in data
    
    def test_workflow_step_to_dict(self):
        """WorkflowStepシリアライゼーションテスト"""
        step = WorkflowStep(
            stage=WorkflowStage.DETECTION,
            name="Test Step",
            description="Test step description",
            module="test_module",
            action="test_action",
            parameters={"param": "value"},
            status="pending"
        )
        
        data = step.to_dict()
        assert isinstance(data, dict)
        assert data["stage"] == "detection"
        assert data["name"] == "Test Step"
        assert data["module"] == "test_module"
        assert data["action"] == "test_action"
        assert data["status"] == "pending"
        assert data["parameters"] == {"param": "value"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
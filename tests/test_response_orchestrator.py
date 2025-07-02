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
    WorkflowRequest,
    IncidentResponse,
    DashboardData,
    WorkflowStage,
    WorkflowStatus,
    AutomationLevel,
    WorkflowTemplate,
    ApprovalRequest
)


class TestResponseOrchestrator:
    """Response Orchestrator基本テスト"""
    
    @pytest.fixture
    def mock_config(self):
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
    def mock_modules(self):
        """依存モジュールのモック"""
        return {
            "slack_reader": Mock(),
            "prometheus_analyzer": Mock(),
            "github_searcher": Mock(),
            "llm_wrapper": Mock(),
            "alert_correlator": Mock(),
            "ssh_executor": Mock()
        }
    
    @pytest.fixture
    def orchestrator(self, mock_config, mock_modules):
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
    
    def test_initialization(self, orchestrator):
        """初期化テスト"""
        assert orchestrator.max_concurrent_workflows == 5
        assert orchestrator.max_steps_per_workflow == 50
        assert orchestrator.approval_timeout_minutes == 60
        assert isinstance(orchestrator.active_workflows, dict)
        assert isinstance(orchestrator.workflow_templates, dict)
        assert isinstance(orchestrator.pending_approvals, dict)
    
    def test_load_default_templates(self, orchestrator):
        """デフォルトテンプレート読み込みテスト"""
        assert len(orchestrator.workflow_templates) >= 2
        
        template_names = list(orchestrator.workflow_templates.keys())
        assert "incident_response" in template_names
        assert "performance_investigation" in template_names
        
        incident_template = orchestrator.workflow_templates["incident_response"]
        assert isinstance(incident_template, WorkflowTemplate)
        assert incident_template.name == "Standard Incident Response"
        assert len(incident_template.stages) >= 6
        assert incident_template.automation_level == AutomationLevel.SEMI_AUTO
    
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
    def sample_incident_event(self):
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
    
    @pytest.fixture
    def sample_workflow_request(self):
        """サンプルワークフローリクエスト"""
        return WorkflowRequest(
            workflow_type="incident_response",
            automation_level=AutomationLevel.SEMI_AUTO,
            priority=1,
            requested_by="sre-engineer",
            context={"incident_severity": "critical"}
        )
    
    def test_create_workflow_instance(self, orchestrator, sample_incident_event, sample_workflow_request):
        """ワークフローインスタンス作成テスト"""
        instance = orchestrator.create_workflow_instance(
            incident_event=sample_incident_event,
            workflow_request=sample_workflow_request
        )
        
        assert isinstance(instance, WorkflowInstance)
        assert instance.workflow_type == "incident_response"
        assert instance.automation_level == AutomationLevel.SEMI_AUTO
        assert instance.priority == 1
        assert instance.status == WorkflowStatus.PENDING
        assert instance.incident_event == sample_incident_event
        assert instance.workflow_request == sample_workflow_request
        assert len(instance.steps) >= 6  # incident_responseテンプレートのステップ数
    
    def test_workflow_steps_creation(self, orchestrator, sample_incident_event, sample_workflow_request):
        """ワークフローステップ作成テスト"""
        instance = orchestrator.create_workflow_instance(
            incident_event=sample_incident_event,
            workflow_request=sample_workflow_request
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
            assert step.workflow_id == instance.workflow_id
            assert step.timeout_minutes > 0
            assert len(step.required_modules) >= 1
    
    @pytest.mark.asyncio
    async def test_start_workflow(self, orchestrator, sample_incident_event, sample_workflow_request):
        """ワークフロー開始テスト"""
        instance = orchestrator.create_workflow_instance(
            incident_event=sample_incident_event,
            workflow_request=sample_workflow_request
        )
        
        # ワークフロー開始
        result = await orchestrator.start_workflow(instance.workflow_id)
        
        assert result is True
        assert instance.workflow_id in orchestrator.active_workflows
        assert instance.status == WorkflowStatus.RUNNING
        assert instance.started_at is not None
    
    @pytest.mark.asyncio
    async def test_start_workflow_not_found(self, orchestrator):
        """存在しないワークフロー開始テスト"""
        result = await orchestrator.start_workflow("nonexistent_id")
        assert result is False
    
    def test_get_workflow_status(self, orchestrator, sample_incident_event, sample_workflow_request):
        """ワークフローステータス取得テスト"""
        instance = orchestrator.create_workflow_instance(
            incident_event=sample_incident_event,
            workflow_request=sample_workflow_request
        )
        
        status = orchestrator.get_workflow_status(instance.workflow_id)
        assert isinstance(status, dict)
        assert status["workflow_id"] == instance.workflow_id
        assert status["status"] == "pending"
        assert status["priority"] == 1
        assert "steps" in status
        assert len(status["steps"]) >= 6
    
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
            workflow_id="test_workflow",
            stage=WorkflowStage.DETECTION,
            name="Alert Detection",
            description="Detect and collect alerts from all sources",
            required_modules=["slack_reader", "prometheus_analyzer"],
            auto_execute=True,
            require_approval=False,
            timeout_minutes=5,
            parameters={"time_window_minutes": 30}
        )
    
    @pytest.fixture
    def sample_execution_step(self):
        """サンプル実行ステップ"""
        return WorkflowStep(
            workflow_id="test_workflow",
            stage=WorkflowStage.EXECUTION,
            name="Execute Commands",
            description="Execute remediation commands",
            required_modules=["ssh_executor"],
            auto_execute=False,
            require_approval=True,
            timeout_minutes=60,
            parameters={"commands": ["systemctl restart app"]}
        )
    
    @pytest.mark.asyncio
    async def test_execute_detection_step(self, orchestrator, sample_detection_step):
        """検知ステップ実行テスト"""
        # モックセットアップ
        orchestrator.slack_reader.get_recent_alerts = AsyncMock(return_value=[])
        orchestrator.prometheus_analyzer.analyze_system_health = AsyncMock(return_value=Mock())
        
        result = await orchestrator.execute_step(sample_detection_step)
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert sample_detection_step.status == WorkflowStatus.COMPLETED
        assert sample_detection_step.completed_at is not None
        assert sample_detection_step.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_execute_step_with_approval(self, orchestrator, sample_execution_step):
        """承認必要ステップ実行テスト"""
        result = await orchestrator.execute_step(sample_execution_step)
        
        assert isinstance(result, dict)
        assert result["success"] is False
        assert result["reason"] == "approval_required"
        assert sample_execution_step.status == WorkflowStatus.WAITING_APPROVAL
        assert sample_execution_step.approval_request_id is not None
    
    @pytest.mark.asyncio
    async def test_execute_step_timeout(self, orchestrator, sample_detection_step):
        """ステップタイムアウトテスト"""
        # 遅延を発生させるモック
        async def slow_operation():
            await asyncio.sleep(0.1)  # テスト用の短い遅延
            return []
        
        orchestrator.slack_reader.get_recent_alerts = slow_operation
        
        # タイムアウトを非常に短く設定
        sample_detection_step.timeout_minutes = 0.001  # 約0.06秒
        
        result = await orchestrator.execute_step(sample_detection_step)
        
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "timeout" in result["reason"].lower()
        assert sample_detection_step.status == WorkflowStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_execute_step_module_error(self, orchestrator, sample_detection_step):
        """モジュールエラーステップ実行テスト"""
        # エラーを発生させるモック
        orchestrator.slack_reader.get_recent_alerts = AsyncMock(side_effect=Exception("Connection failed"))
        
        result = await orchestrator.execute_step(sample_detection_step)
        
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error" in result
        assert sample_detection_step.status == WorkflowStatus.FAILED


class TestApprovalManagement:
    """承認管理テスト"""
    
    @pytest.fixture
    def sample_approval_request(self):
        """サンプル承認リクエスト"""
        return ApprovalRequest(
            workflow_id="test_workflow",
            step_id="test_step",
            approval_level="level_2",
            required_approvers=["sre-team", "team-lead"],
            request_reason="Critical command execution required",
            context={"command": "systemctl restart app"},
            expiry_time=datetime.now() + timedelta(hours=1)
        )
    
    def test_create_approval_request(self, orchestrator, sample_approval_request):
        """承認リクエスト作成テスト"""
        approval_id = orchestrator.create_approval_request(sample_approval_request)
        
        assert approval_id is not None
        assert approval_id in orchestrator.pending_approvals
        assert orchestrator.pending_approvals[approval_id] == sample_approval_request
        assert sample_approval_request.status == WorkflowStatus.WAITING_APPROVAL
    
    @pytest.mark.asyncio
    async def test_approve_request_success(self, orchestrator, sample_approval_request):
        """承認成功テスト"""
        approval_id = orchestrator.create_approval_request(sample_approval_request)
        
        # 1人目の承認
        result1 = await orchestrator.approve_request(approval_id, "sre-team", True, "Approved by SRE")
        assert result1 is True
        assert "sre-team" in sample_approval_request.approvals
        assert sample_approval_request.status == WorkflowStatus.WAITING_APPROVAL  # まだ不十分
        
        # 2人目の承認で完了
        result2 = await orchestrator.approve_request(approval_id, "team-lead", True, "Approved by team lead")
        assert result2 is True
        assert "team-lead" in sample_approval_request.approvals
        assert sample_approval_request.status == WorkflowStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_approve_request_rejection(self, orchestrator, sample_approval_request):
        """承認拒否テスト"""
        approval_id = orchestrator.create_approval_request(sample_approval_request)
        
        # 拒否
        result = await orchestrator.approve_request(approval_id, "team-lead", False, "Too risky")
        assert result is True
        assert sample_approval_request.status == WorkflowStatus.FAILED
        assert sample_approval_request.approvals["team-lead"]["decision"] is False
    
    @pytest.mark.asyncio
    async def test_approve_request_not_found(self, orchestrator):
        """存在しない承認リクエストテスト"""
        result = await orchestrator.approve_request("nonexistent", "approver", True, "comment")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_approve_request_expired(self, orchestrator):
        """期限切れ承認リクエストテスト"""
        expired_request = ApprovalRequest(
            workflow_id="test",
            step_id="test",
            approval_level="level_1",
            required_approvers=["sre-team"],
            request_reason="Test",
            expiry_time=datetime.now() - timedelta(minutes=1)  # 期限切れ
        )
        
        approval_id = orchestrator.create_approval_request(expired_request)
        
        result = await orchestrator.approve_request(approval_id, "sre-team", True, "Late approval")
        assert result is False
        assert expired_request.status == WorkflowStatus.FAILED
    
    def test_get_pending_approvals(self, orchestrator, sample_approval_request):
        """承認待ちリスト取得テスト"""
        # 有効な承認リクエスト追加
        approval_id = orchestrator.create_approval_request(sample_approval_request)
        
        # 期限切れの承認リクエスト追加
        expired_request = ApprovalRequest(
            workflow_id="expired",
            step_id="expired",
            approval_level="level_1",
            required_approvers=["sre-team"],
            request_reason="Expired test",
            expiry_time=datetime.now() - timedelta(minutes=1)
        )
        expired_id = orchestrator.create_approval_request(expired_request)
        
        pending = orchestrator.get_pending_approvals()
        
        assert len(pending) == 1  # 期限切れは除外される
        assert pending[0] == sample_approval_request
        assert expired_id not in orchestrator.pending_approvals  # 期限切れは削除される


class TestDashboardGeneration:
    """ダッシュボード生成テスト"""
    
    def test_generate_dashboard_data(self, orchestrator):
        """ダッシュボードデータ生成テスト"""
        # サンプルワークフロー追加
        sample_event = IncidentEvent(
            source="test",
            event_type="alert",
            severity="critical",
            title="Test incident"
        )
        sample_request = WorkflowRequest(
            workflow_type="incident_response",
            automation_level=AutomationLevel.SEMI_AUTO
        )
        
        instance = orchestrator.create_workflow_instance(sample_event, sample_request)
        instance.status = WorkflowStatus.RUNNING
        
        # ダッシュボードデータ生成
        dashboard = orchestrator.generate_dashboard_data()
        
        assert isinstance(dashboard, DashboardData)
        assert dashboard.active_workflows_count == 1
        assert dashboard.pending_approvals_count == 0
        assert len(dashboard.recent_workflows) == 1
        assert dashboard.system_health["orchestrator_status"] == "healthy"
        
        # ワークフローサマリー確認
        workflow_summary = dashboard.recent_workflows[0]
        assert workflow_summary["workflow_id"] == instance.workflow_id
        assert workflow_summary["status"] == "running"
        assert workflow_summary["priority"] == instance.priority
    
    def test_dashboard_metrics_calculation(self, orchestrator):
        """ダッシュボードメトリクス計算テスト"""
        # 複数のワークフローを作成
        for i in range(3):
            event = IncidentEvent(source="test", event_type="alert", severity="medium", title=f"Test {i}")
            request = WorkflowRequest(workflow_type="incident_response")
            instance = orchestrator.create_workflow_instance(event, request)
            
            if i == 0:
                instance.status = WorkflowStatus.COMPLETED
                instance.completed_at = datetime.now()
                instance.started_at = datetime.now() - timedelta(minutes=30)
            elif i == 1:
                instance.status = WorkflowStatus.RUNNING
                instance.started_at = datetime.now() - timedelta(minutes=10)
            else:
                instance.status = WorkflowStatus.PENDING
        
        dashboard = orchestrator.generate_dashboard_data()
        
        assert dashboard.active_workflows_count == 2  # RUNNING + PENDING
        assert dashboard.completed_workflows_count == 1
        assert dashboard.average_resolution_time_minutes == 30.0
        assert dashboard.workflow_success_rate == 1.0  # 1 completed, 0 failed
    
    def test_dashboard_system_health(self, orchestrator):
        """ダッシュボードシステムヘルス確認テスト"""
        dashboard = orchestrator.generate_dashboard_data()
        
        health = dashboard.system_health
        assert "orchestrator_status" in health
        assert "total_modules_available" in health
        assert "active_modules_count" in health
        assert "memory_usage_mb" in health
        assert "uptime_minutes" in health
        
        assert health["orchestrator_status"] == "healthy"
        assert health["total_modules_available"] == 6  # 依存モジュール数


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
        
        # ワークフローリクエスト作成
        workflow_request = WorkflowRequest(
            workflow_type="incident_response",
            automation_level=AutomationLevel.SEMI_AUTO,
            priority=1,
            requested_by="on-call-engineer"
        )
        
        # ワークフローインスタンス作成
        instance = orchestrator.create_workflow_instance(incident_event, workflow_request)
        
        # 自動実行可能ステップのモック設定
        orchestrator.slack_reader.get_recent_alerts = AsyncMock(return_value=[])
        orchestrator.prometheus_analyzer.analyze_system_health = AsyncMock(return_value=Mock())
        orchestrator.alert_correlator.process_incident_workflow = AsyncMock(return_value={})
        orchestrator.llm_wrapper.analyze_incident = AsyncMock(return_value=Mock())
        orchestrator.github_searcher.search_similar_issues = AsyncMock(return_value=[])
        
        # ワークフロー開始
        start_result = await orchestrator.start_workflow(instance.workflow_id)
        assert start_result is True
        
        # ステータス確認
        status = orchestrator.get_workflow_status(instance.workflow_id)
        assert status["status"] == "running"
        
        # 自動実行ステップが完了していることを確認
        completed_stages = [step["stage"] for step in status["steps"] if step["status"] == "completed"]
        assert "detection" in completed_stages
        assert "correlation" in completed_stages
        assert "analysis" in completed_stages
        
        # 承認待ちステップが存在することを確認
        waiting_steps = [step for step in status["steps"] if step["status"] == "waiting_approval"]
        assert len(waiting_steps) >= 1
    
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
        
        # 自動監視ワークフローリクエスト
        workflow_request = WorkflowRequest(
            workflow_type="performance_investigation",
            automation_level=AutomationLevel.AUTO_MONITOR,
            priority=3,
            requested_by="monitoring-system"
        )
        
        # モック設定
        orchestrator.prometheus_analyzer.analyze_system_health = AsyncMock(return_value=Mock())
        orchestrator.prometheus_analyzer.get_performance_metrics = AsyncMock(return_value=Mock())
        orchestrator.llm_wrapper.analyze_performance = AsyncMock(return_value=Mock())
        orchestrator.github_searcher.search_similar_issues = AsyncMock(return_value=[])
        
        # ワークフロー作成・開始
        instance = orchestrator.create_workflow_instance(performance_event, workflow_request)
        start_result = await orchestrator.start_workflow(instance.workflow_id)
        
        assert start_result is True
        
        # AUTO_MONITORレベルでは自動実行される
        status = orchestrator.get_workflow_status(instance.workflow_id)
        completed_steps = [step for step in status["steps"] if step["status"] == "completed"]
        assert len(completed_steps) >= 3  # 多くのステップが自動完了
    
    def test_workflow_template_customization(self, orchestrator):
        """ワークフローテンプレートカスタマイズテスト"""
        # カスタムテンプレート作成
        custom_template = WorkflowTemplate(
            name="Custom Security Audit",
            description="Custom security audit workflow",
            workflow_type="security_audit",
            automation_level=AutomationLevel.MANUAL,
            stages=[
                WorkflowStage.DETECTION,
                WorkflowStage.ANALYSIS,
                WorkflowStage.COMPLETION
            ],
            estimated_duration_minutes=90
        )
        
        # テンプレート登録
        orchestrator.register_workflow_template(custom_template)
        
        assert "security_audit" in orchestrator.workflow_templates
        assert orchestrator.workflow_templates["security_audit"] == custom_template
        
        # カスタムテンプレートでワークフロー作成
        event = IncidentEvent(source="security", event_type="audit", severity="info", title="Security Audit")
        request = WorkflowRequest(workflow_type="security_audit", automation_level=AutomationLevel.MANUAL)
        
        instance = orchestrator.create_workflow_instance(event, request)
        assert instance.workflow_type == "security_audit"
        assert len(instance.steps) == 3  # カスタムステージ数


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
        request = WorkflowRequest(workflow_type="test", automation_level=AutomationLevel.MANUAL)
        
        instance = WorkflowInstance(
            workflow_type="test_workflow",
            automation_level=AutomationLevel.SEMI_AUTO,
            priority=2,
            incident_event=event,
            workflow_request=request,
            status=WorkflowStatus.RUNNING
        )
        
        data = instance.to_dict()
        assert isinstance(data, dict)
        assert data["workflow_type"] == "test_workflow"
        assert data["automation_level"] == "semi_auto"
        assert data["priority"] == 2
        assert data["status"] == "running"
        assert "steps" in data
    
    def test_dashboard_data_to_dict(self):
        """DashboardDataシリアライゼーションテスト"""
        dashboard = DashboardData(
            active_workflows_count=5,
            pending_approvals_count=2,
            completed_workflows_count=10,
            failed_workflows_count=1,
            average_resolution_time_minutes=45.5,
            workflow_success_rate=0.91,
            recent_workflows=[],
            system_health={"status": "healthy"},
            performance_metrics={"cpu": 50.0}
        )
        
        data = dashboard.to_dict()
        assert isinstance(data, dict)
        assert data["active_workflows_count"] == 5
        assert data["pending_approvals_count"] == 2
        assert data["workflow_success_rate"] == 0.91
        assert data["system_health"]["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
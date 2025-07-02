#!/usr/bin/env python3
"""
AALS Module 9: Response Orchestrator
全モジュール統合制御・ワークフロー管理モジュール

PURPOSE: 全てのAALSモジュールを統合し、エンドツーエンドのSREワークフローを制御する。
         人間の承認インターフェース、リアルタイムダッシュボード、自動化レベル調整を提供。
DEPENDENCIES: Module 2-8 (全モジュール)
INPUT: IncidentEvent, WorkflowRequest
OUTPUT: IncidentResponse, WorkflowStatus, Dashboard
INTEGRATION: 最上位制御モジュール
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path

from aals.core.config import get_config_manager
from aals.core.logger import get_logger, AuditAction, AuditLogEntry, audit_log

# 依存モジュール
from aals.modules.slack_alert_reader import SlackAlertReader, SlackMessage
from aals.modules.prometheus_analyzer import PrometheusAnalyzer, SystemHealthReport
from aals.modules.github_issues_searcher import GitHubIssuesSearcher, GitHubSearchReport
from aals.modules.llm_wrapper import LLMWrapper, LLMResponse
from aals.modules.alert_correlator import (
    AlertCorrelator, CorrelatedAlert, IntegratedRecommendation, 
    EscalationDecision, EscalationLevel
)
from aals.modules.ssh_executor import (
    SSHExecutor, CommandRequest, ExecutionReport, PermissionLevel,
    SSHTarget, ApprovalStatus
)

logger = get_logger(__name__)


class WorkflowStage(Enum):
    """ワークフロー段階"""
    DETECTION = "detection"              # 異常検知
    CORRELATION = "correlation"          # 相関分析
    ANALYSIS = "analysis"               # 詳細分析
    PLANNING = "planning"               # 対応計画
    APPROVAL = "approval"               # 承認プロセス
    EXECUTION = "execution"             # 実行
    MONITORING = "monitoring"           # 実行監視
    COMPLETION = "completion"           # 完了


class WorkflowStatus(Enum):
    """ワークフロー状態"""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"


class AutomationLevel(Enum):
    """自動化レベル"""
    MANUAL = "manual"                   # 完全手動
    SEMI_AUTO = "semi_auto"            # 半自動（承認必要）
    AUTO_MONITOR = "auto_monitor"      # 自動実行+監視
    FULL_AUTO = "full_auto"            # 完全自動


@dataclass
class IncidentEvent:
    """インシデントイベント"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""                    # slack, prometheus, github, manual
    event_type: str = ""               # alert, metric_threshold, issue_created
    severity: str = "info"             # critical, high, medium, low, info
    title: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "event_type": self.event_type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }


@dataclass
class WorkflowStep:
    """ワークフロー実行ステップ"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage: WorkflowStage = WorkflowStage.DETECTION
    name: str = ""
    description: str = ""
    module: str = ""                   # 実行モジュール名
    action: str = ""                   # 実行アクション
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"           # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "stage": self.stage.value,
            "name": self.name,
            "description": self.description,
            "module": self.module,
            "action": self.action,
            "parameters": self.parameters,
            "status": self.status,
            "result": self.result,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": self.execution_time
        }


@dataclass
class WorkflowInstance:
    """ワークフロー実行インスタンス"""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    incident_event: IncidentEvent = None
    automation_level: AutomationLevel = AutomationLevel.SEMI_AUTO
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_stage: WorkflowStage = WorkflowStage.DETECTION
    steps: List[WorkflowStep] = field(default_factory=list)
    approvals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "incident_event": self.incident_event.to_dict() if self.incident_event else None,
            "automation_level": self.automation_level.value,
            "status": self.status.value,
            "current_stage": self.current_stage.value,
            "steps": [step.to_dict() for step in self.steps],
            "approvals": self.approvals,
            "execution_context": self.execution_context,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by": self.created_by
        }


@dataclass
class IncidentResponse:
    """インシデント対応結果"""
    incident_id: str
    workflow_id: str
    status: str
    summary: str
    actions_taken: List[Dict[str, Any]]
    recommendations: List[str]
    metrics: Dict[str, Any]
    duration_minutes: float
    automation_level: str
    human_interventions: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "summary": self.summary,
            "actions_taken": self.actions_taken,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
            "duration_minutes": self.duration_minutes,
            "automation_level": self.automation_level,
            "human_interventions": self.human_interventions,
            "timestamp": self.timestamp.isoformat()
        }


class WorkflowTemplate:
    """ワークフローテンプレート"""
    
    @staticmethod
    def create_standard_incident_workflow() -> List[WorkflowStep]:
        """標準インシデント対応ワークフロー"""
        return [
            WorkflowStep(
                stage=WorkflowStage.DETECTION,
                name="Alert Collection",
                description="Collect alerts from multiple sources",
                module="alert_correlator",
                action="collect_alert_contexts",
                parameters={"time_window_minutes": 30}
            ),
            WorkflowStep(
                stage=WorkflowStage.CORRELATION,
                name="Alert Correlation",
                description="Analyze correlations between alerts",
                module="alert_correlator",
                action="analyze_correlations",
                parameters={}
            ),
            WorkflowStep(
                stage=WorkflowStage.ANALYSIS,
                name="System Health Analysis",
                description="Analyze current system health",
                module="prometheus_analyzer",
                action="analyze_system_health",
                parameters={"hours_back": 1}
            ),
            WorkflowStep(
                stage=WorkflowStage.ANALYSIS,
                name="Similar Issues Search",
                description="Search for similar past issues",
                module="github_searcher",
                action="search_similar_incidents",
                parameters={"max_results": 5}
            ),
            WorkflowStep(
                stage=WorkflowStage.ANALYSIS,
                name="LLM Analysis",
                description="AI-powered incident analysis",
                module="llm_wrapper",
                action="analyze_incident",
                parameters={}
            ),
            WorkflowStep(
                stage=WorkflowStage.PLANNING,
                name="Generate Recommendations",
                description="Generate integrated recommendations",
                module="alert_correlator",
                action="generate_integrated_recommendations",
                parameters={}
            ),
            WorkflowStep(
                stage=WorkflowStage.PLANNING,
                name="Escalation Decision",
                description="Determine escalation requirements",
                module="alert_correlator",
                action="make_escalation_decision",
                parameters={}
            ),
            WorkflowStep(
                stage=WorkflowStage.EXECUTION,
                name="Execute Commands",
                description="Execute recommended commands",
                module="ssh_executor",
                action="execute_command",
                parameters={"dry_run": True}  # デフォルトはdry-run
            ),
            WorkflowStep(
                stage=WorkflowStage.MONITORING,
                name="Post-Execution Monitoring",
                description="Monitor system after execution",
                module="prometheus_analyzer",
                action="analyze_system_health",
                parameters={"hours_back": 0.5}
            )
        ]
    
    @staticmethod
    def create_performance_investigation_workflow() -> List[WorkflowStep]:
        """パフォーマンス調査ワークフロー"""
        return [
            WorkflowStep(
                stage=WorkflowStage.DETECTION,
                name="Performance Metrics Collection",
                description="Collect performance-related metrics",
                module="prometheus_analyzer",
                action="analyze_system_health",
                parameters={"hours_back": 2}
            ),
            WorkflowStep(
                stage=WorkflowStage.ANALYSIS,
                name="Trend Analysis",
                description="Analyze performance trends",
                module="prometheus_analyzer",
                action="analyze_metric_trend",
                parameters={}
            ),
            WorkflowStep(
                stage=WorkflowStage.ANALYSIS,
                name="Performance Issue Search",
                description="Search for similar performance issues",
                module="github_searcher",
                action="search_similar_incidents",
                parameters={"keywords": ["performance", "slow", "timeout"]}
            ),
            WorkflowStep(
                stage=WorkflowStage.PLANNING,
                name="Performance Recommendations",
                description="Generate performance improvement recommendations",
                module="llm_wrapper",
                action="analyze_metrics",
                parameters={}
            )
        ]


class ResponseOrchestrator:
    """レスポンスオーケストレーター"""
    
    def __init__(self):
        """初期化"""
        self.config_manager = get_config_manager()
        self.config = self.config_manager.get_module_config("response_orchestrator")
        
        # 依存モジュール初期化
        self.slack_reader = SlackAlertReader()
        self.prometheus_analyzer = PrometheusAnalyzer()
        self.github_searcher = GitHubIssuesSearcher()
        self.llm_wrapper = LLMWrapper()
        self.alert_correlator = AlertCorrelator()
        self.ssh_executor = SSHExecutor()
        
        # ワークフロー設定
        self.default_automation_level = AutomationLevel(
            self.config.config.get("default_automation_level", "semi_auto")
        )
        self.max_concurrent_workflows = self.config.config.get("max_concurrent_workflows", 10)
        self.workflow_timeout_minutes = self.config.config.get("workflow_timeout_minutes", 60)
        
        # アクティブワークフロー
        self.active_workflows: Dict[str, WorkflowInstance] = {}
        self.workflow_history: List[WorkflowInstance] = []
        
        # モジュールマッピング
        self.modules = {
            "slack_reader": self.slack_reader,
            "prometheus_analyzer": self.prometheus_analyzer,
            "github_searcher": self.github_searcher,
            "llm_wrapper": self.llm_wrapper,
            "alert_correlator": self.alert_correlator,
            "ssh_executor": self.ssh_executor
        }
        
        # 承認者設定
        self.approvers = self.config.config.get("approvers", {})
        
        logger.info("Response Orchestrator initialized",
                   automation_level=self.default_automation_level.value,
                   max_workflows=self.max_concurrent_workflows)
    
    async def start(self):
        """オーケストレーター開始"""
        # 依存モジュール開始
        await self.ssh_executor.start()
        
        logger.info("Response Orchestrator started")
    
    async def stop(self):
        """オーケストレーター停止"""
        # アクティブワークフローのクリーンアップ
        for workflow_id in list(self.active_workflows.keys()):
            await self.cancel_workflow(workflow_id, "System shutdown")
        
        # 依存モジュール停止
        await self.ssh_executor.stop()
        
        logger.info("Response Orchestrator stopped")
    
    async def verify_setup(self) -> bool:
        """セットアップ確認"""
        try:
            # 全依存モジュールの確認
            module_status = {}
            for name, module in self.modules.items():
                if hasattr(module, 'verify_setup'):
                    status = await module.verify_setup()
                    module_status[name] = status
                else:
                    module_status[name] = True
            
            successful_modules = sum(1 for status in module_status.values() if status)
            total_modules = len(module_status)
            
            logger.info("Response Orchestrator setup verification completed",
                       successful_modules=successful_modules,
                       total_modules=total_modules,
                       module_status=module_status)
            
            # 半数以上のモジュールが正常であれば動作可能
            return successful_modules >= total_modules // 2
            
        except Exception as e:
            logger.error("Response Orchestrator setup verification failed",
                        error=str(e), exception_type=type(e).__name__)
            return False
    
    async def handle_incident(
        self,
        incident_event: IncidentEvent,
        automation_level: Optional[AutomationLevel] = None,
        workflow_template: Optional[str] = None
    ) -> WorkflowInstance:
        """インシデント処理"""
        
        # 自動化レベル決定
        automation_level = automation_level or self._determine_automation_level(incident_event)
        
        # ワークフローテンプレート選択
        workflow_steps = self._get_workflow_template(workflow_template, incident_event)
        
        # ワークフローインスタンス作成
        workflow = WorkflowInstance(
            incident_event=incident_event,
            automation_level=automation_level,
            steps=workflow_steps,
            status=WorkflowStatus.PENDING
        )
        
        # アクティブワークフローに追加
        self.active_workflows[workflow.workflow_id] = workflow
        
        # 監査ログ
        audit_log(AuditLogEntry(
            action=AuditAction.CREATE,
            resource=f"incident_workflow:{workflow.workflow_id}",
            result="success",
            details=f"Incident: {incident_event.title}, Automation: {automation_level.value}",
            risk_level=self._get_risk_level(incident_event.severity)
        ))
        
        logger.info("Incident workflow created",
                   workflow_id=workflow.workflow_id,
                   incident_id=incident_event.event_id,
                   automation_level=automation_level.value,
                   steps_count=len(workflow_steps))
        
        # ワークフロー開始
        asyncio.create_task(self._execute_workflow(workflow))
        
        return workflow
    
    def _determine_automation_level(self, incident_event: IncidentEvent) -> AutomationLevel:
        """自動化レベル決定"""
        severity = incident_event.severity.lower()
        source = incident_event.source.lower()
        
        # 重要度ベースの判定
        if severity == "critical":
            # クリティカルは手動承認必須
            return AutomationLevel.SEMI_AUTO
        elif severity == "high":
            return AutomationLevel.SEMI_AUTO
        elif severity == "medium":
            return AutomationLevel.AUTO_MONITOR
        else:
            return AutomationLevel.FULL_AUTO
    
    def _get_workflow_template(
        self, 
        template_name: Optional[str], 
        incident_event: IncidentEvent
    ) -> List[WorkflowStep]:
        """ワークフローテンプレート取得"""
        if template_name == "performance":
            return WorkflowTemplate.create_performance_investigation_workflow()
        else:
            # デフォルトは標準インシデント対応
            return WorkflowTemplate.create_standard_incident_workflow()
    
    def _get_risk_level(self, severity: str) -> str:
        """リスクレベル変換"""
        risk_map = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low",
            "info": "low"
        }
        return risk_map.get(severity.lower(), "medium")
    
    async def _execute_workflow(self, workflow: WorkflowInstance):
        """ワークフロー実行"""
        try:
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
            
            logger.info("Workflow execution started",
                       workflow_id=workflow.workflow_id)
            
            for step in workflow.steps:
                # ステップ実行前チェック
                if workflow.status in [WorkflowStatus.CANCELLED, WorkflowStatus.FAILED]:
                    break
                
                # 現在のステージ更新
                workflow.current_stage = step.stage
                
                # 承認が必要なステップかチェック
                if self._requires_approval(workflow, step):
                    workflow.status = WorkflowStatus.WAITING_APPROVAL
                    await self._request_approval(workflow, step)
                    
                    # 承認待ち
                    approval_granted = await self._wait_for_approval(workflow, step)
                    if not approval_granted:
                        workflow.status = WorkflowStatus.FAILED
                        break
                    
                    workflow.status = WorkflowStatus.RUNNING
                
                # ステップ実行
                success = await self._execute_step(workflow, step)
                
                if not success:
                    workflow.status = WorkflowStatus.FAILED
                    break
            
            # 完了処理
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = datetime.now()
                
                # 完了後の後処理
                await self._post_execution_processing(workflow)
            
            # ワークフロー履歴に移動
            self._archive_workflow(workflow)
            
            logger.info("Workflow execution completed",
                       workflow_id=workflow.workflow_id,
                       status=workflow.status.value,
                       duration=f"{(workflow.completed_at - workflow.started_at).total_seconds():.1f}s" if workflow.completed_at else "N/A")
            
        except Exception as e:
            logger.error("Workflow execution failed",
                        workflow_id=workflow.workflow_id,
                        error=str(e),
                        exception_type=type(e).__name__)
            
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now()
            self._archive_workflow(workflow)
    
    def _requires_approval(self, workflow: WorkflowInstance, step: WorkflowStep) -> bool:
        """承認が必要かチェック"""
        # 自動化レベルによる判定
        if workflow.automation_level == AutomationLevel.MANUAL:
            return True
        elif workflow.automation_level == AutomationLevel.FULL_AUTO:
            return False
        
        # 実行ステップは承認必要
        if step.stage == WorkflowStage.EXECUTION:
            if step.module == "ssh_executor":
                return True
        
        # 高リスクアクションは承認必要
        high_risk_actions = ["execute_command", "restart_service", "scale_resources"]
        if step.action in high_risk_actions:
            return True
        
        return False
    
    async def _request_approval(self, workflow: WorkflowInstance, step: WorkflowStep):
        """承認リクエスト"""
        approval_id = str(uuid.uuid4())
        
        # 必要な承認者決定
        required_approvers = self._get_required_approvers(workflow, step)
        
        # 承認リクエスト記録
        workflow.approvals[approval_id] = {
            "step_id": step.step_id,
            "required_approvers": required_approvers,
            "approvals": {},
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=30)).isoformat()
        }
        
        logger.info("Approval requested",
                   workflow_id=workflow.workflow_id,
                   step_id=step.step_id,
                   approval_id=approval_id,
                   required_approvers=required_approvers)
    
    def _get_required_approvers(self, workflow: WorkflowInstance, step: WorkflowStep) -> List[str]:
        """必要承認者取得"""
        severity = workflow.incident_event.severity.lower()
        
        if step.stage == WorkflowStage.EXECUTION:
            if severity == "critical":
                return self.approvers.get("critical", ["sre-lead", "on-call"])
            elif severity == "high":
                return self.approvers.get("high", ["sre-team"])
            else:
                return self.approvers.get("medium", ["sre-team"])
        
        return self.approvers.get("default", ["sre-team"])
    
    async def _wait_for_approval(self, workflow: WorkflowInstance, step: WorkflowStep) -> bool:
        """承認待ち"""
        timeout_minutes = 30
        check_interval = 5  # 5秒間隔でチェック
        max_checks = (timeout_minutes * 60) // check_interval
        
        for _ in range(max_checks):
            await asyncio.sleep(check_interval)
            
            # 承認状況チェック
            for approval_id, approval_data in workflow.approvals.items():
                if approval_data["step_id"] == step.step_id:
                    if approval_data["status"] == "approved":
                        return True
                    elif approval_data["status"] == "rejected":
                        return False
            
            # ワークフローがキャンセルされた場合
            if workflow.status == WorkflowStatus.CANCELLED:
                return False
        
        # タイムアウト
        logger.warning("Approval timeout",
                      workflow_id=workflow.workflow_id,
                      step_id=step.step_id)
        return False
    
    async def _execute_step(self, workflow: WorkflowInstance, step: WorkflowStep) -> bool:
        """ステップ実行"""
        step.status = "running"
        step.started_at = datetime.now()
        
        logger.info("Executing workflow step",
                   workflow_id=workflow.workflow_id,
                   step_id=step.step_id,
                   module=step.module,
                   action=step.action)
        
        try:
            # モジュール取得
            module = self.modules.get(step.module)
            if not module:
                raise ValueError(f"Module not found: {step.module}")
            
            # アクション実行
            result = await self._call_module_action(
                module, 
                step.action, 
                step.parameters,
                workflow.execution_context
            )
            
            # 結果を実行コンテキストに保存
            workflow.execution_context[f"{step.module}_{step.action}"] = result
            
            step.result = result
            step.status = "completed"
            step.completed_at = datetime.now()
            step.execution_time = (step.completed_at - step.started_at).total_seconds()
            
            logger.info("Workflow step completed",
                       workflow_id=workflow.workflow_id,
                       step_id=step.step_id,
                       execution_time=f"{step.execution_time:.2f}s")
            
            return True
            
        except Exception as e:
            step.status = "failed"
            step.error_message = str(e)
            step.completed_at = datetime.now()
            step.execution_time = (step.completed_at - step.started_at).total_seconds()
            
            logger.error("Workflow step failed",
                        workflow_id=workflow.workflow_id,
                        step_id=step.step_id,
                        error=str(e),
                        exception_type=type(e).__name__)
            
            return False
    
    async def _call_module_action(
        self, 
        module: Any, 
        action: str, 
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """モジュールアクション呼び出し"""
        
        # コンテキストからパラメータを補完
        enhanced_params = self._enhance_parameters(parameters, context)
        
        # アクション実行
        if action == "collect_alert_contexts":
            return await module.collect_alert_contexts(**enhanced_params)
        elif action == "analyze_correlations":
            contexts = context.get("alert_correlator_collect_alert_contexts", [])
            return module.analyze_correlations(contexts)
        elif action == "analyze_system_health":
            return await module.analyze_system_health(**enhanced_params)
        elif action == "search_similar_incidents":
            # 前のステップから説明文を取得
            description = self._extract_incident_description(context)
            enhanced_params["description"] = description
            return await module.search_similar_incidents(**enhanced_params)
        elif action == "analyze_incident":
            description = self._extract_incident_description(context)
            return await module.analyze_incident(description)
        elif action == "generate_integrated_recommendations":
            correlation = context.get("alert_correlator_analyze_correlations", [])
            if correlation:
                return await module.generate_integrated_recommendations(correlation[0])
            return None
        elif action == "make_escalation_decision":
            correlation = context.get("alert_correlator_analyze_correlations", [])
            if correlation:
                return module.make_escalation_decision(correlation[0])
            return None
        elif action == "execute_command":
            # SSH実行は特別処理
            return await self._execute_ssh_commands(module, enhanced_params, context)
        else:
            # 汎用アクション実行
            if hasattr(module, action):
                method = getattr(module, action)
                if asyncio.iscoroutinefunction(method):
                    return await method(**enhanced_params)
                else:
                    return method(**enhanced_params)
            else:
                raise ValueError(f"Action not found: {action} on module {type(module).__name__}")
    
    def _enhance_parameters(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータをコンテキストで拡張"""
        enhanced = parameters.copy()
        
        # コンテキストから必要な値を自動補完
        # 例: 時間窓、相関ID、メトリクス名など
        
        return enhanced
    
    def _extract_incident_description(self, context: Dict[str, Any]) -> str:
        """コンテキストからインシデント説明を抽出"""
        descriptions = []
        
        # 相関分析結果から
        correlations = context.get("alert_correlator_analyze_correlations", [])
        if correlations:
            descriptions.append(f"Correlated incident with {len(correlations)} alerts")
        
        # システムヘルス分析から
        health_report = context.get("prometheus_analyzer_analyze_system_health")
        if health_report:
            descriptions.append(f"System health: {health_report.overall_health}")
        
        return "; ".join(descriptions) if descriptions else "System incident detected"
    
    async def _execute_ssh_commands(
        self, 
        ssh_executor: SSHExecutor, 
        parameters: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionReport:
        """SSH コマンド実行"""
        
        # 推奨アクションから実行コマンドを抽出
        recommendations = context.get("alert_correlator_generate_integrated_recommendations")
        
        if not recommendations:
            # デフォルトコマンド
            command = "systemctl status"
            targets = [SSHTarget(host="localhost")]
        else:
            # 推奨アクションから最初のコマンドを実行
            immediate_actions = recommendations.immediate_actions
            if immediate_actions:
                command = immediate_actions[0]  # 簡易的にテキストをそのまま使用
                targets = [SSHTarget(host="localhost")]  # 実環境では適切なターゲット決定
            else:
                command = "echo 'No actions recommended'"
                targets = [SSHTarget(host="localhost")]
        
        # 実行
        return await ssh_executor.execute_command(
            command=command,
            targets=targets,
            permission_level=PermissionLevel.READ_ONLY,
            dry_run=parameters.get("dry_run", True),
            requested_by="orchestrator"
        )
    
    async def _post_execution_processing(self, workflow: WorkflowInstance):
        """実行後処理"""
        # 結果サマリー生成
        summary = self._generate_workflow_summary(workflow)
        
        # 監査ログ
        audit_log(AuditLogEntry(
            action=AuditAction.COMPLETE,
            resource=f"incident_workflow:{workflow.workflow_id}",
            result="success",
            details=summary,
            risk_level=self._get_risk_level(workflow.incident_event.severity)
        ))
        
        logger.info("Workflow post-processing completed",
                   workflow_id=workflow.workflow_id,
                   summary=summary[:100])
    
    def _generate_workflow_summary(self, workflow: WorkflowInstance) -> str:
        """ワークフローサマリー生成"""
        completed_steps = [s for s in workflow.steps if s.status == "completed"]
        failed_steps = [s for s in workflow.steps if s.status == "failed"]
        total_time = (workflow.completed_at - workflow.started_at).total_seconds() if workflow.completed_at else 0
        
        return (f"Workflow completed: {len(completed_steps)}/{len(workflow.steps)} steps successful, "
                f"{len(failed_steps)} failed, duration: {total_time:.1f}s")
    
    def _archive_workflow(self, workflow: WorkflowInstance):
        """ワークフロー アーカイブ"""
        if workflow.workflow_id in self.active_workflows:
            del self.active_workflows[workflow.workflow_id]
        
        self.workflow_history.append(workflow)
        
        # 履歴サイズ制限
        max_history = self.config.config.get("max_workflow_history", 1000)
        if len(self.workflow_history) > max_history:
            self.workflow_history = self.workflow_history[-max_history:]
    
    async def approve_workflow_step(
        self, 
        workflow_id: str, 
        approval_id: str, 
        approver: str, 
        decision: bool
    ) -> bool:
        """ワークフローステップ承認"""
        
        if workflow_id not in self.active_workflows:
            logger.warning("Workflow not found for approval",
                          workflow_id=workflow_id)
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        if approval_id not in workflow.approvals:
            logger.warning("Approval request not found",
                          workflow_id=workflow_id,
                          approval_id=approval_id)
            return False
        
        approval_data = workflow.approvals[approval_id]
        
        # 承認記録
        approval_data["approvals"][approver] = {
            "decision": decision,
            "timestamp": datetime.now().isoformat()
        }
        
        # 拒否の場合
        if not decision:
            approval_data["status"] = "rejected"
            logger.info("Workflow step rejected",
                       workflow_id=workflow_id,
                       approval_id=approval_id,
                       approver=approver)
            return True
        
        # 全承認者の承認確認
        required_approvers = approval_data["required_approvers"]
        approvals = approval_data["approvals"]
        
        if len(approvals) >= len(required_approvers):
            approval_data["status"] = "approved"
            logger.info("Workflow step approved",
                       workflow_id=workflow_id,
                       approval_id=approval_id,
                       approvers=list(approvals.keys()))
        
        return True
    
    async def cancel_workflow(self, workflow_id: str, reason: str = "User cancelled") -> bool:
        """ワークフローキャンセル"""
        
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.now()
        
        # 実行中のステップを停止
        for step in workflow.steps:
            if step.status == "running":
                step.status = "cancelled"
                step.completed_at = datetime.now()
        
        logger.info("Workflow cancelled",
                   workflow_id=workflow_id,
                   reason=reason)
        
        self._archive_workflow(workflow)
        return True
    
    def get_active_workflows(self) -> List[WorkflowInstance]:
        """アクティブワークフロー取得"""
        return list(self.active_workflows.values())
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """ワークフロー状態取得"""
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]
        
        # 履歴からも検索
        for workflow in self.workflow_history:
            if workflow.workflow_id == workflow_id:
                return workflow
        
        return None
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """承認待ちリスト取得"""
        pending = []
        
        for workflow in self.active_workflows.values():
            if workflow.status == WorkflowStatus.WAITING_APPROVAL:
                for approval_id, approval_data in workflow.approvals.items():
                    if approval_data["status"] == "pending":
                        pending.append({
                            "workflow_id": workflow.workflow_id,
                            "approval_id": approval_id,
                            "incident_title": workflow.incident_event.title,
                            "step_name": next(
                                (s.name for s in workflow.steps if s.step_id == approval_data["step_id"]), 
                                "Unknown Step"
                            ),
                            "required_approvers": approval_data["required_approvers"],
                            "created_at": approval_data["created_at"],
                            "expires_at": approval_data["expires_at"]
                        })
        
        return pending
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """ダッシュボードデータ生成"""
        now = datetime.now()
        
        # アクティブワークフロー統計
        active_count = len(self.active_workflows)
        waiting_approval_count = len([w for w in self.active_workflows.values() 
                                    if w.status == WorkflowStatus.WAITING_APPROVAL])
        
        # 最近の履歴統計
        recent_history = [w for w in self.workflow_history 
                         if w.completed_at and (now - w.completed_at).days < 7]
        
        completed_count = len([w for w in recent_history if w.status == WorkflowStatus.COMPLETED])
        failed_count = len([w for w in recent_history if w.status == WorkflowStatus.FAILED])
        
        # 平均実行時間
        execution_times = []
        for workflow in recent_history:
            if workflow.started_at and workflow.completed_at:
                duration = (workflow.completed_at - workflow.started_at).total_seconds()
                execution_times.append(duration)
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "timestamp": now.isoformat(),
            "active_workflows": {
                "total": active_count,
                "waiting_approval": waiting_approval_count,
                "running": active_count - waiting_approval_count
            },
            "recent_history": {
                "completed": completed_count,
                "failed": failed_count,
                "success_rate": completed_count / (completed_count + failed_count) if (completed_count + failed_count) > 0 else 0
            },
            "performance": {
                "avg_execution_time_seconds": avg_execution_time,
                "total_workflows_processed": len(recent_history)
            },
            "pending_approvals": len(self.get_pending_approvals())
        }


# モジュール使用例
async def main():
    """使用例"""
    orchestrator = ResponseOrchestrator()
    
    # 開始
    await orchestrator.start()
    
    try:
        # セットアップ確認
        if not await orchestrator.verify_setup():
            print("❌ Setup verification failed")
            return
        
        # サンプルインシデント
        incident = IncidentEvent(
            source="prometheus",
            event_type="metric_threshold",
            severity="high",
            title="High API Response Time",
            description="API response time exceeded 5 seconds threshold",
            metadata={"metric": "api_response_time", "value": 6.2, "threshold": 5.0}
        )
        
        # インシデント処理開始
        workflow = await orchestrator.handle_incident(
            incident,
            automation_level=AutomationLevel.SEMI_AUTO
        )
        
        print(f"📊 Incident Workflow Started")
        print(f"   Workflow ID: {workflow.workflow_id}")
        print(f"   Automation Level: {workflow.automation_level.value}")
        print(f"   Steps: {len(workflow.steps)}")
        
        # ダッシュボードデータ
        dashboard = orchestrator.generate_dashboard_data()
        print(f"\n📈 Dashboard")
        print(f"   Active Workflows: {dashboard['active_workflows']['total']}")
        print(f"   Pending Approvals: {dashboard['pending_approvals']}")
        
        # 承認待ちリスト
        pending = orchestrator.get_pending_approvals()
        print(f"\n⏳ Pending Approvals: {len(pending)}")
        
    finally:
        # 停止
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
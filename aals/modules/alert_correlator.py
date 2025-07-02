#!/usr/bin/env python3
"""
AALS Module 7: Alert Correlator
アラート相関分析・統合推奨生成モジュール

PURPOSE: 複数のアラート源からの情報を統合し、相関分析を行い、統合された対応策を生成する
DEPENDENCIES: Module 2(Slack), Module 4(Prometheus), Module 5(GitHub), Module 6(LLM)
INPUT: SlackMessage, PrometheusMetric, GitHubIssue, LLMAnalysis
OUTPUT: CorrelatedAlert, IntegratedRecommendation, EscalationDecision
"""

import asyncio
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from aals.core.config import get_config_manager
from aals.core.logger import get_logger, AuditAction, AuditLogEntry, audit_log
from aals.modules.slack_alert_reader import SlackAlertReader, SlackMessage
from aals.modules.prometheus_analyzer import PrometheusAnalyzer, AlertEvent, AlertSeverity
from aals.modules.github_issues_searcher import GitHubIssuesSearcher, SimilarIssue
from aals.modules.llm_wrapper import LLMWrapper, IncidentAnalysisResult

logger = get_logger(__name__)


class IncidentSeverity(Enum):
    """インシデント重要度"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EscalationLevel(Enum):
    """エスカレーションレベル"""
    AUTO_RESOLVE = "auto_resolve"      # 自動解決
    MONITOR_ONLY = "monitor_only"      # 監視のみ
    HUMAN_REVIEW = "human_review"      # 人間確認
    IMMEDIATE_ACTION = "immediate_action"  # 即座対応
    EMERGENCY = "emergency"            # 緊急事態


@dataclass
class AlertContext:
    """アラートコンテキスト情報"""
    source: str
    timestamp: datetime
    severity: str
    content: Dict[str, Any]
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "content": self.content,
            "confidence": self.confidence
        }


@dataclass
class CorrelationRule:
    """相関分析ルール"""
    name: str
    conditions: List[Dict[str, Any]]
    weight: float
    time_window_minutes: int
    description: str
    enabled: bool = True


@dataclass
class CorrelatedAlert:
    """相関分析されたアラート"""
    correlation_id: str
    primary_source: str
    related_sources: List[str]
    severity: IncidentSeverity
    confidence_score: float
    time_window: Tuple[datetime, datetime]
    alert_contexts: List[AlertContext]
    correlation_evidence: Dict[str, Any]
    estimated_impact: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "primary_source": self.primary_source,
            "related_sources": self.related_sources,
            "severity": self.severity.value,
            "confidence_score": self.confidence_score,
            "time_window": [self.time_window[0].isoformat(), self.time_window[1].isoformat()],
            "alert_contexts": [ctx.to_dict() for ctx in self.alert_contexts],
            "correlation_evidence": self.correlation_evidence,
            "estimated_impact": self.estimated_impact,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass  
class IntegratedRecommendation:
    """統合推奨アクション"""
    correlation_id: str
    priority: int
    immediate_actions: List[str]
    investigation_steps: List[str]
    mitigation_strategies: List[str]
    preventive_measures: List[str]
    estimated_resolution_time: int  # minutes
    required_skills: List[str]
    risk_assessment: Dict[str, Any]
    automation_candidates: List[str]
    escalation_triggers: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "immediate_actions": self.immediate_actions,
            "investigation_steps": self.investigation_steps,
            "mitigation_strategies": self.mitigation_strategies,
            "preventive_measures": self.preventive_measures,
            "estimated_resolution_time": self.estimated_resolution_time,
            "required_skills": self.required_skills,
            "risk_assessment": self.risk_assessment,
            "automation_candidates": self.automation_candidates,
            "escalation_triggers": self.escalation_triggers,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EscalationDecision:
    """エスカレーション判定"""
    correlation_id: str
    escalation_level: EscalationLevel
    reasoning: str
    required_approvals: List[str]
    time_limit_minutes: Optional[int]
    notification_targets: List[str]
    automated_actions_allowed: bool
    human_intervention_required: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "escalation_level": self.escalation_level.value,
            "reasoning": self.reasoning,
            "required_approvals": self.required_approvals,
            "time_limit_minutes": self.time_limit_minutes,
            "notification_targets": self.notification_targets,
            "automated_actions_allowed": self.automated_actions_allowed,
            "human_intervention_required": self.human_intervention_required,
            "timestamp": self.timestamp.isoformat()
        }


class AlertCorrelator:
    """アラート相関分析器"""
    
    def __init__(self):
        """初期化"""
        self.config_manager = get_config_manager()
        self.config = self.config_manager.get_module_config("alert_correlator")
        
        # 依存モジュール初期化
        self.slack_reader = SlackAlertReader()
        self.prometheus_analyzer = PrometheusAnalyzer()
        self.github_searcher = GitHubIssuesSearcher()
        self.llm_wrapper = LLMWrapper()
        
        # 相関分析設定
        self.correlation_window_minutes = self.config.config.get("correlation_window_minutes", 30)
        self.confidence_threshold = self.config.config.get("confidence_threshold", 0.7)
        self.max_correlations_per_window = self.config.config.get("max_correlations_per_window", 10)
        
        # 相関ルール読み込み
        self._load_correlation_rules()
        
        # アクティブな相関追跡
        self.active_correlations: Dict[str, CorrelatedAlert] = {}
        
        logger.info("Alert Correlator initialized",
                   correlation_window=self.correlation_window_minutes,
                   confidence_threshold=self.confidence_threshold,
                   rules_count=len(self.correlation_rules))
    
    def _load_correlation_rules(self):
        """相関分析ルールを読み込み"""
        rules_config = self.config.config.get("correlation_rules", [])
        
        # デフォルトルール
        default_rules = [
            CorrelationRule(
                name="prometheus_slack_correlation",
                conditions=[
                    {"source": "prometheus", "severity": ["critical", "warning"]},
                    {"source": "slack", "keywords": ["api", "response", "error", "down"]}
                ],
                weight=0.8,
                time_window_minutes=15,
                description="Prometheus alerts correlating with Slack notifications"
            ),
            CorrelationRule(
                name="github_issue_pattern",
                conditions=[
                    {"source": "github", "similar_issues": {"min_count": 1}},
                    {"source": "prometheus", "trend": "increasing"}
                ],
                weight=0.6,
                time_window_minutes=60,
                description="GitHub issue patterns matching current metrics trends"
            ),
            CorrelationRule(
                name="multi_source_critical",
                conditions=[
                    {"source": "slack", "alert_level": "critical"},
                    {"source": "prometheus", "severity": "critical"},
                    {"source": "llm", "confidence": {"min": 0.8}}
                ],
                weight=0.9,
                time_window_minutes=10,
                description="Critical alerts from multiple sources"
            )
        ]
        
        self.correlation_rules = []
        
        # 設定ファイルのルールを追加
        for rule_config in rules_config:
            rule = CorrelationRule(**rule_config)
            self.correlation_rules.append(rule)
        
        # デフォルトルールを追加（設定にない場合）
        for default_rule in default_rules:
            if not any(r.name == default_rule.name for r in self.correlation_rules):
                self.correlation_rules.append(default_rule)
        
        logger.info("Correlation rules loaded", 
                   total_rules=len(self.correlation_rules),
                   enabled_rules=len([r for r in self.correlation_rules if r.enabled]))
    
    async def verify_setup(self) -> bool:
        """セットアップ確認"""
        try:
            # 依存モジュールのセットアップ確認
            modules_status = {
                "slack": await self.slack_reader.verify_setup(),
                "prometheus": await self.prometheus_analyzer.verify_setup(),
                "github": await self.github_searcher.verify_setup(),
                "llm": await self.llm_wrapper.verify_setup()
            }
            
            success_count = sum(1 for status in modules_status.values() if status)
            total_count = len(modules_status)
            
            logger.info("Alert Correlator setup verification completed",
                       successful_modules=success_count,
                       total_modules=total_count,
                       modules_status=modules_status)
            
            # 最低2つのモジュールが正常であれば動作可能
            return success_count >= 2
            
        except Exception as e:
            logger.error("Alert Correlator setup verification failed",
                        error=str(e), exception_type=type(e).__name__)
            return False
    
    async def collect_alert_contexts(
        self, 
        time_window_minutes: int = None
    ) -> List[AlertContext]:
        """アラートコンテキスト収集"""
        time_window_minutes = time_window_minutes or self.correlation_window_minutes
        contexts = []
        
        try:
            # Slackアラート収集
            slack_messages = await self.slack_reader.get_recent_alerts(
                hours_back=time_window_minutes // 60 + 1
            )
            
            for msg in slack_messages:
                if self._is_within_time_window(msg.datetime, time_window_minutes):
                    context = AlertContext(
                        source="slack",
                        timestamp=msg.datetime,
                        severity=msg.alert_level or "info",
                        content={
                            "message": msg.text,
                            "channel": msg.channel_name,
                            "user": msg.user,
                            "keywords": getattr(msg, 'keywords', []),
                            "url": msg.message_url
                        },
                        confidence=0.8 if msg.is_alert else 0.3
                    )
                    contexts.append(context)
            
            logger.info("Slack contexts collected", count=len([c for c in contexts if c.source == "slack"]))
            
        except Exception as e:
            logger.warning("Failed to collect Slack contexts", error=str(e))
        
        try:
            # Prometheusアラート収集
            system_report = await self.prometheus_analyzer.analyze_system_health(
                hours_back=time_window_minutes // 60 + 1
            )
            
            for analysis in system_report.metric_analyses:
                for alert in analysis.alerts:
                    if self._is_within_time_window(alert.timestamp, time_window_minutes):
                        context = AlertContext(
                            source="prometheus",
                            timestamp=alert.timestamp,
                            severity=alert.severity.value,
                            content={
                                "metric_name": alert.metric_name,
                                "current_value": alert.current_value,
                                "threshold_value": alert.threshold_value,
                                "message": alert.message,
                                "labels": alert.labels,
                                "trend": analysis.trend,
                                "anomaly_score": analysis.anomaly_score
                            },
                            confidence=0.9
                        )
                        contexts.append(context)
            
            logger.info("Prometheus contexts collected", count=len([c for c in contexts if c.source == "prometheus"]))
            
        except Exception as e:
            logger.warning("Failed to collect Prometheus contexts", error=str(e))
        
        return contexts
    
    def _is_within_time_window(self, timestamp: datetime, window_minutes: int) -> bool:
        """時間窓内判定"""
        now = datetime.now()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=None)
        if now.tzinfo is None:
            now = now.replace(tzinfo=None)
            
        time_diff = abs((now - timestamp).total_seconds() / 60)
        return time_diff <= window_minutes
    
    def analyze_correlations(self, contexts: List[AlertContext]) -> List[CorrelatedAlert]:
        """相関分析実行"""
        correlations = []
        
        # 時間でソート
        contexts_sorted = sorted(contexts, key=lambda x: x.timestamp)
        
        for rule in self.correlation_rules:
            if not rule.enabled:
                continue
                
            matches = self._find_rule_matches(contexts_sorted, rule)
            
            for match in matches:
                correlation = self._create_correlation(match, rule)
                if correlation.confidence_score >= self.confidence_threshold:
                    correlations.append(correlation)
        
        # 重複排除・優先度ソート
        correlations = self._deduplicate_correlations(correlations)
        correlations.sort(key=lambda x: (x.severity.value, -x.confidence_score))
        
        logger.info("Correlation analysis completed",
                   total_contexts=len(contexts),
                   correlations_found=len(correlations),
                   avg_confidence=statistics.mean([c.confidence_score for c in correlations]) if correlations else 0)
        
        return correlations[:self.max_correlations_per_window]
    
    def _find_rule_matches(self, contexts: List[AlertContext], rule: CorrelationRule) -> List[List[AlertContext]]:
        """ルールマッチング"""
        matches = []
        
        # 条件に合うコンテキストを収集
        condition_matches = {}
        for i, condition in enumerate(rule.conditions):
            condition_matches[i] = []
            
            for context in contexts:
                if self._context_matches_condition(context, condition):
                    condition_matches[i].append(context)
        
        # 全条件を満たす組み合わせを検索
        if all(condition_matches.values()):
            # 時間窓内での組み合わせを作成
            for primary_context in condition_matches[0]:
                match_group = [primary_context]
                
                for i in range(1, len(rule.conditions)):
                    for candidate in condition_matches[i]:
                        if self._contexts_within_window(
                            primary_context, candidate, rule.time_window_minutes
                        ):
                            match_group.append(candidate)
                            break
                
                if len(match_group) >= len(rule.conditions):
                    matches.append(match_group)
        
        return matches
    
    def _context_matches_condition(self, context: AlertContext, condition: Dict[str, Any]) -> bool:
        """コンテキスト条件マッチング"""
        # ソースチェック
        if "source" in condition and context.source != condition["source"]:
            return False
        
        # 重要度チェック
        if "severity" in condition:
            severities = condition["severity"]
            if isinstance(severities, str):
                severities = [severities]
            if context.severity not in severities:
                return False
        
        # キーワードチェック
        if "keywords" in condition:
            keywords = condition["keywords"]
            content_text = str(context.content).lower()
            if not any(keyword.lower() in content_text for keyword in keywords):
                return False
        
        # 信頼度チェック
        if "confidence" in condition:
            conf_req = condition["confidence"]
            if isinstance(conf_req, dict):
                if "min" in conf_req and context.confidence < conf_req["min"]:
                    return False
                if "max" in conf_req and context.confidence > conf_req["max"]:
                    return False
            elif isinstance(conf_req, (int, float)):
                if context.confidence < conf_req:
                    return False
        
        return True
    
    def _contexts_within_window(
        self, 
        context1: AlertContext, 
        context2: AlertContext, 
        window_minutes: int
    ) -> bool:
        """コンテキスト間時間窓チェック"""
        time_diff = abs((context1.timestamp - context2.timestamp).total_seconds() / 60)
        return time_diff <= window_minutes
    
    def _create_correlation(self, matched_contexts: List[AlertContext], rule: CorrelationRule) -> CorrelatedAlert:
        """相関アラート作成"""
        correlation_id = f"corr_{int(datetime.now().timestamp())}_{rule.name}"
        
        # 重要度計算
        max_severity = max(matched_contexts, key=lambda x: self._severity_to_numeric(x.severity))
        severity = self._numeric_to_incident_severity(self._severity_to_numeric(max_severity.severity))
        
        # 信頼度計算
        confidence = statistics.mean([ctx.confidence for ctx in matched_contexts]) * rule.weight
        
        # 時間窓計算
        timestamps = [ctx.timestamp for ctx in matched_contexts]
        time_window = (min(timestamps), max(timestamps))
        
        # 影響度推定
        estimated_impact = self._estimate_impact(matched_contexts)
        
        # 相関証拠
        correlation_evidence = {
            "rule_name": rule.name,
            "rule_description": rule.description,
            "matched_conditions": len(rule.conditions),
            "time_span_minutes": (time_window[1] - time_window[0]).total_seconds() / 60,
            "sources_involved": list(set(ctx.source for ctx in matched_contexts))
        }
        
        return CorrelatedAlert(
            correlation_id=correlation_id,
            primary_source=matched_contexts[0].source,
            related_sources=list(set(ctx.source for ctx in matched_contexts[1:])),
            severity=severity,
            confidence_score=min(confidence, 1.0),
            time_window=time_window,
            alert_contexts=matched_contexts,
            correlation_evidence=correlation_evidence,
            estimated_impact=estimated_impact
        )
    
    def _severity_to_numeric(self, severity: str) -> int:
        """重要度を数値に変換"""
        severity_map = {
            "critical": 5,
            "high": 4,
            "warning": 3,
            "medium": 3,
            "low": 2,
            "info": 1
        }
        return severity_map.get(severity.lower(), 1)
    
    def _numeric_to_incident_severity(self, numeric: int) -> IncidentSeverity:
        """数値を重要度に変換"""
        if numeric >= 5:
            return IncidentSeverity.CRITICAL
        elif numeric >= 4:
            return IncidentSeverity.HIGH
        elif numeric >= 3:
            return IncidentSeverity.MEDIUM
        elif numeric >= 2:
            return IncidentSeverity.LOW
        else:
            return IncidentSeverity.INFO
    
    def _estimate_impact(self, contexts: List[AlertContext]) -> Dict[str, Any]:
        """影響度推定"""
        impact = {
            "affected_systems": set(),
            "user_impact_level": "unknown",
            "business_impact": "unknown",
            "estimated_affected_users": 0,
            "service_availability": "unknown"
        }
        
        for context in contexts:
            content = context.content
            
            # システム影響度
            if context.source == "prometheus":
                impact["affected_systems"].add(content.get("metric_name", "unknown"))
            elif context.source == "slack":
                if any(keyword in content.get("message", "").lower() 
                       for keyword in ["api", "database", "server", "service"]):
                    impact["affected_systems"].add("application_service")
            
            # ユーザー影響度推定
            if context.severity in ["critical", "high"]:
                impact["user_impact_level"] = "high"
                impact["estimated_affected_users"] = max(
                    impact["estimated_affected_users"], 100
                )
        
        impact["affected_systems"] = list(impact["affected_systems"])
        return impact
    
    def _deduplicate_correlations(self, correlations: List[CorrelatedAlert]) -> List[CorrelatedAlert]:
        """相関の重複排除"""
        # 簡単な重複排除（同じソース組み合わせ + 時間近接）
        unique_correlations = []
        
        for correlation in correlations:
            is_duplicate = False
            
            for existing in unique_correlations:
                if (set(correlation.related_sources) == set(existing.related_sources) and
                    abs((correlation.timestamp - existing.timestamp).total_seconds()) < 300):  # 5分以内
                    # より高い信頼度を保持
                    if correlation.confidence_score > existing.confidence_score:
                        unique_correlations.remove(existing)
                        unique_correlations.append(correlation)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_correlations.append(correlation)
        
        return unique_correlations
    
    async def generate_integrated_recommendations(
        self, 
        correlation: CorrelatedAlert
    ) -> IntegratedRecommendation:
        """統合推奨アクション生成"""
        
        # LLMを使った詳細分析
        llm_analysis = await self._get_llm_analysis(correlation)
        
        # GitHub類似ケース検索
        similar_cases = await self._find_similar_cases(correlation)
        
        # 推奨アクション生成
        recommendations = self._generate_recommendations(correlation, llm_analysis, similar_cases)
        
        logger.info("Integrated recommendations generated",
                   correlation_id=correlation.correlation_id,
                   immediate_actions=len(recommendations.immediate_actions),
                   mitigation_strategies=len(recommendations.mitigation_strategies))
        
        return recommendations
    
    async def _get_llm_analysis(self, correlation: CorrelatedAlert) -> Optional[IncidentAnalysisResult]:
        """LLM分析取得"""
        try:
            # 相関情報を結合した説明文作成
            incident_description = self._create_incident_description(correlation)
            
            analysis = await self.llm_wrapper.analyze_incident(incident_description)
            return analysis
            
        except Exception as e:
            logger.warning("LLM analysis failed for correlation",
                          correlation_id=correlation.correlation_id,
                          error=str(e))
            return None
    
    def _create_incident_description(self, correlation: CorrelatedAlert) -> str:
        """インシデント説明文作成"""
        description_parts = [
            f"Correlated incident detected with {correlation.severity.value} severity",
            f"Primary source: {correlation.primary_source}",
            f"Related sources: {', '.join(correlation.related_sources)}",
            f"Time window: {correlation.time_window[0].strftime('%H:%M')} - {correlation.time_window[1].strftime('%H:%M')}",
            ""
        ]
        
        for i, context in enumerate(correlation.alert_contexts):
            description_parts.append(f"Alert {i+1} ({context.source}):")
            if context.source == "slack":
                description_parts.append(f"  Message: {context.content.get('message', '')}")
            elif context.source == "prometheus":
                description_parts.append(f"  Metric: {context.content.get('metric_name')} = {context.content.get('current_value')}")
                description_parts.append(f"  Threshold: {context.content.get('threshold_value')}")
            description_parts.append("")
        
        return "\n".join(description_parts)
    
    async def _find_similar_cases(self, correlation: CorrelatedAlert) -> List[SimilarIssue]:
        """類似ケース検索"""
        try:
            # 検索キーワード抽出
            keywords = []
            for context in correlation.alert_contexts:
                if context.source == "slack":
                    keywords.extend(context.content.get("keywords", []))
                elif context.source == "prometheus":
                    keywords.append(context.content.get("metric_name", ""))
            
            keywords = [k for k in keywords if k and len(k) > 2][:5]  # 上位5キーワード
            
            if keywords:
                description = f"incident {' '.join(keywords)} {correlation.severity.value}"
                search_report = await self.github_searcher.search_similar_incidents(
                    description, max_results=5
                )
                return search_report.similar_issues
            
        except Exception as e:
            logger.warning("Similar cases search failed",
                          correlation_id=correlation.correlation_id,
                          error=str(e))
        
        return []
    
    def _generate_recommendations(
        self,
        correlation: CorrelatedAlert,
        llm_analysis: Optional[IncidentAnalysisResult],
        similar_cases: List[SimilarIssue]
    ) -> IntegratedRecommendation:
        """推奨アクション生成"""
        
        # 基本推奨アクション
        immediate_actions = [
            "Acknowledge the incident and start investigation",
            "Check system health dashboard for additional context",
            "Verify if the issue is affecting users"
        ]
        
        investigation_steps = [
            "Review correlated alerts and their timelines",
            "Check for recent deployments or configuration changes",
            "Analyze metric trends for related systems"
        ]
        
        mitigation_strategies = [
            "Monitor the situation for escalation",
            "Prepare rollback procedures if needed",
            "Scale resources if performance issue is confirmed"
        ]
        
        preventive_measures = [
            "Review alerting thresholds for false positives",
            "Improve monitoring coverage for early detection",
            "Document incident for future correlation patterns"
        ]
        
        # LLM分析結果を統合
        if llm_analysis:
            immediate_actions.extend(llm_analysis.mitigation_steps[:3])
            mitigation_strategies.extend(llm_analysis.mitigation_steps[3:])
            preventive_measures.extend(llm_analysis.prevention_strategies)
        
        # 類似ケースからの学習
        if similar_cases:
            for case in similar_cases[:2]:
                if case.issue.resolution_time:
                    hours = case.issue.resolution_time.total_seconds() / 3600
                    investigation_steps.append(
                        f"Check similar issue #{case.issue.number} (resolved in {hours:.1f}h)"
                    )
        
        # 重複排除
        immediate_actions = list(dict.fromkeys(immediate_actions))
        investigation_steps = list(dict.fromkeys(investigation_steps))
        mitigation_strategies = list(dict.fromkeys(mitigation_strategies))
        preventive_measures = list(dict.fromkeys(preventive_measures))
        
        # 推定解決時間
        estimated_time = self._estimate_resolution_time(correlation, similar_cases)
        
        # 必要スキル
        required_skills = self._identify_required_skills(correlation)
        
        # リスク評価
        risk_assessment = self._assess_risks(correlation)
        
        # 自動化候補
        automation_candidates = self._identify_automation_candidates(correlation)
        
        # エスカレーション条件
        escalation_triggers = self._define_escalation_triggers(correlation)
        
        return IntegratedRecommendation(
            correlation_id=correlation.correlation_id,
            priority=self._calculate_priority(correlation),
            immediate_actions=immediate_actions,
            investigation_steps=investigation_steps,
            mitigation_strategies=mitigation_strategies,
            preventive_measures=preventive_measures,
            estimated_resolution_time=estimated_time,
            required_skills=required_skills,
            risk_assessment=risk_assessment,
            automation_candidates=automation_candidates,
            escalation_triggers=escalation_triggers
        )
    
    def _estimate_resolution_time(self, correlation: CorrelatedAlert, similar_cases: List[SimilarIssue]) -> int:
        """解決時間推定（分）"""
        base_time = {
            IncidentSeverity.CRITICAL: 60,
            IncidentSeverity.HIGH: 120,
            IncidentSeverity.MEDIUM: 240,
            IncidentSeverity.LOW: 480,
            IncidentSeverity.INFO: 1440
        }
        
        estimated = base_time.get(correlation.severity, 240)
        
        # 類似ケースからの調整
        if similar_cases:
            case_times = []
            for case in similar_cases:
                if case.issue.resolution_time:
                    case_times.append(case.issue.resolution_time.total_seconds() / 60)
            
            if case_times:
                avg_case_time = statistics.mean(case_times)
                estimated = int((estimated + avg_case_time) / 2)
        
        return estimated
    
    def _identify_required_skills(self, correlation: CorrelatedAlert) -> List[str]:
        """必要スキル特定"""
        skills = set()
        
        for context in correlation.alert_contexts:
            if context.source == "prometheus":
                skills.add("Infrastructure Monitoring")
                metric_name = context.content.get("metric_name", "")
                if "cpu" in metric_name or "memory" in metric_name:
                    skills.add("System Performance")
                elif "http" in metric_name or "api" in metric_name:
                    skills.add("Application Performance")
            elif context.source == "slack":
                skills.add("Incident Communication")
                message = context.content.get("message", "").lower()
                if "database" in message:
                    skills.add("Database Administration")
                elif "network" in message:
                    skills.add("Network Engineering")
        
        return list(skills)
    
    def _assess_risks(self, correlation: CorrelatedAlert) -> Dict[str, Any]:
        """リスク評価"""
        return {
            "data_loss_risk": "low" if correlation.severity != IncidentSeverity.CRITICAL else "medium",
            "service_degradation": "possible" if len(correlation.related_sources) > 1 else "unlikely",
            "user_impact": correlation.estimated_impact.get("user_impact_level", "unknown"),
            "escalation_probability": min(0.8, correlation.confidence_score + 0.2),
            "automated_recovery_possible": correlation.confidence_score > 0.8 and correlation.severity in [IncidentSeverity.LOW, IncidentSeverity.MEDIUM]
        }
    
    def _identify_automation_candidates(self, correlation: CorrelatedAlert) -> List[str]:
        """自動化候補特定"""
        candidates = []
        
        if correlation.severity in [IncidentSeverity.LOW, IncidentSeverity.MEDIUM]:
            candidates.append("Automated alert acknowledgment")
            
            for context in correlation.alert_contexts:
                if context.source == "prometheus":
                    metric = context.content.get("metric_name", "")
                    if "cpu" in metric or "memory" in metric:
                        candidates.append("Resource scaling automation")
                    elif "disk" in metric:
                        candidates.append("Log cleanup automation")
        
        if correlation.confidence_score > 0.8:
            candidates.append("Automated runbook execution")
        
        return candidates
    
    def _define_escalation_triggers(self, correlation: CorrelatedAlert) -> List[str]:
        """エスカレーション条件定義"""
        triggers = []
        
        if correlation.severity == IncidentSeverity.CRITICAL:
            triggers.append("Immediate escalation to on-call engineer")
        
        triggers.extend([
            f"No progress after {self._estimate_resolution_time(correlation, []) // 2} minutes",
            "Additional critical alerts in same time window",
            "User impact confirmed and increasing"
        ])
        
        if correlation.confidence_score < 0.7:
            triggers.append("Unable to determine root cause within 30 minutes")
        
        return triggers
    
    def _calculate_priority(self, correlation: CorrelatedAlert) -> int:
        """優先度計算（1-5、1が最高）"""
        base_priority = {
            IncidentSeverity.CRITICAL: 1,
            IncidentSeverity.HIGH: 2,
            IncidentSeverity.MEDIUM: 3,
            IncidentSeverity.LOW: 4,
            IncidentSeverity.INFO: 5
        }
        
        priority = base_priority.get(correlation.severity, 3)
        
        # 信頼度による調整
        if correlation.confidence_score > 0.9:
            priority = max(1, priority - 1)
        elif correlation.confidence_score < 0.7:
            priority = min(5, priority + 1)
        
        return priority
    
    def make_escalation_decision(self, correlation: CorrelatedAlert) -> EscalationDecision:
        """エスカレーション判定"""
        
        # 基本エスカレーションレベル決定
        if correlation.severity == IncidentSeverity.CRITICAL:
            if correlation.confidence_score > 0.8:
                escalation_level = EscalationLevel.IMMEDIATE_ACTION
            else:
                escalation_level = EscalationLevel.HUMAN_REVIEW
        elif correlation.severity == IncidentSeverity.HIGH:
            escalation_level = EscalationLevel.HUMAN_REVIEW
        elif correlation.severity in [IncidentSeverity.MEDIUM, IncidentSeverity.LOW]:
            if correlation.confidence_score > 0.8:
                escalation_level = EscalationLevel.MONITOR_ONLY
            else:
                escalation_level = EscalationLevel.AUTO_RESOLVE
        else:
            escalation_level = EscalationLevel.AUTO_RESOLVE
        
        # 複数ソース相関での調整
        if len(correlation.related_sources) >= 2:
            if escalation_level == EscalationLevel.AUTO_RESOLVE:
                escalation_level = EscalationLevel.MONITOR_ONLY
            elif escalation_level == EscalationLevel.MONITOR_ONLY:
                escalation_level = EscalationLevel.HUMAN_REVIEW
        
        # 判定理由
        reasoning = f"Severity: {correlation.severity.value}, Confidence: {correlation.confidence_score:.2f}, Sources: {len(correlation.related_sources)+1}"
        
        # 承認要件
        required_approvals = []
        if escalation_level == EscalationLevel.IMMEDIATE_ACTION:
            required_approvals = ["on-call-engineer"]
        elif escalation_level == EscalationLevel.HUMAN_REVIEW:
            required_approvals = ["sre-team-lead"]
        
        # 制限時間
        time_limits = {
            EscalationLevel.EMERGENCY: 5,
            EscalationLevel.IMMEDIATE_ACTION: 15,
            EscalationLevel.HUMAN_REVIEW: 60,
            EscalationLevel.MONITOR_ONLY: None,
            EscalationLevel.AUTO_RESOLVE: None
        }
        
        # 通知対象
        notification_targets = []
        if escalation_level in [EscalationLevel.IMMEDIATE_ACTION, EscalationLevel.EMERGENCY]:
            notification_targets = ["#incidents", "@oncall", "sre-team"]
        elif escalation_level == EscalationLevel.HUMAN_REVIEW:
            notification_targets = ["#alerts", "sre-team"]
        
        return EscalationDecision(
            correlation_id=correlation.correlation_id,
            escalation_level=escalation_level,
            reasoning=reasoning,
            required_approvals=required_approvals,
            time_limit_minutes=time_limits[escalation_level],
            notification_targets=notification_targets,
            automated_actions_allowed=escalation_level in [EscalationLevel.AUTO_RESOLVE, EscalationLevel.MONITOR_ONLY],
            human_intervention_required=escalation_level in [EscalationLevel.HUMAN_REVIEW, EscalationLevel.IMMEDIATE_ACTION, EscalationLevel.EMERGENCY]
        )
    
    async def process_incident_workflow(
        self, 
        time_window_minutes: int = None
    ) -> Dict[str, Any]:
        """インシデントワークフロー処理"""
        
        audit_log(AuditLogEntry(
            action=AuditAction.EXECUTE,
            resource="alert_correlation_workflow",
            result="started",
            details=f"Processing incident workflow with {time_window_minutes or self.correlation_window_minutes}min window"
        ))
        
        try:
            # 1. アラートコンテキスト収集
            contexts = await self.collect_alert_contexts(time_window_minutes)
            
            # 2. 相関分析
            correlations = self.analyze_correlations(contexts)
            
            # 3. 各相関に対する処理
            workflow_results = []
            
            for correlation in correlations:
                # 統合推奨生成
                recommendations = await self.generate_integrated_recommendations(correlation)
                
                # エスカレーション判定
                escalation = self.make_escalation_decision(correlation)
                
                # アクティブ相関として追跡開始
                self.active_correlations[correlation.correlation_id] = correlation
                
                result = {
                    "correlation": correlation.to_dict(),
                    "recommendations": recommendations.to_dict(),
                    "escalation": escalation.to_dict()
                }
                workflow_results.append(result)
                
                logger.info("Incident workflow processed",
                           correlation_id=correlation.correlation_id,
                           severity=correlation.severity.value,
                           escalation_level=escalation.escalation_level.value)
            
            # 監査ログ記録
            audit_log(AuditLogEntry(
                action=AuditAction.EXECUTE,
                resource="alert_correlation_workflow",
                result="success",
                details=f"Processed {len(correlations)} correlations from {len(contexts)} contexts"
            ))
            
            return {
                "timestamp": datetime.now().isoformat(),
                "contexts_analyzed": len(contexts),
                "correlations_found": len(correlations),
                "workflow_results": workflow_results,
                "active_correlations": len(self.active_correlations)
            }
            
        except Exception as e:
            logger.error("Incident workflow processing failed",
                        error=str(e), exception_type=type(e).__name__)
            
            audit_log(AuditLogEntry(
                action=AuditAction.EXECUTE,
                resource="alert_correlation_workflow",
                result="error",
                details=f"Workflow failed: {str(e)}",
                risk_level="medium"
            ))
            
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "correlations_found": 0,
                "workflow_results": []
            }


# モジュール使用例
async def main():
    """使用例"""
    correlator = AlertCorrelator()
    
    # セットアップ確認
    if not await correlator.verify_setup():
        print("❌ Setup verification failed")
        return
    
    # インシデントワークフロー実行
    results = await correlator.process_incident_workflow(time_window_minutes=30)
    
    print(f"📊 Incident Workflow Results")
    print(f"   Contexts analyzed: {results['contexts_analyzed']}")
    print(f"   Correlations found: {results['correlations_found']}")
    print(f"   Active correlations: {results.get('active_correlations', 0)}")
    
    for i, result in enumerate(results.get('workflow_results', [])):
        correlation = result['correlation']
        escalation = result['escalation']
        print(f"\n🔗 Correlation {i+1}:")
        print(f"   ID: {correlation['correlation_id']}")
        print(f"   Severity: {correlation['severity']}")
        print(f"   Confidence: {correlation['confidence_score']:.2f}")
        print(f"   Escalation: {escalation['escalation_level']}")


if __name__ == "__main__":
    asyncio.run(main())
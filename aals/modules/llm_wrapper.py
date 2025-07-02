"""
LLM Wrapper Module

このモジュールは複数のLLMプロバイダー（Claude、OpenAI等）を統一インターフェースで利用するための
ラッパー機能を提供します。SREタスクに特化したプロンプトテンプレートと推論機能を含みます。
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import os

from aals.core.config import get_config_manager
from aals.core.logger import get_logger
from aals.integrations.claude_client import (
    ClaudeAPIClient, ClaudeAPIConfig, ClaudeMessage, ClaudeResponse, ClaudeModel
)

logger = get_logger(__name__)


class LLMProvider(Enum):
    """LLMプロバイダー種別"""
    CLAUDE = "claude"
    OPENAI = "openai"
    AZURE = "azure"


class PromptTemplate(Enum):
    """プロンプトテンプレート種別"""
    INCIDENT_ANALYSIS = "incident_analysis"
    METRIC_ANALYSIS = "metric_analysis"
    SOLUTION_GENERATION = "solution_generation"
    ALERT_CORRELATION = "alert_correlation"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"


@dataclass
class LLMRequest:
    """LLMリクエスト"""
    prompt: str
    template: Optional[PromptTemplate] = None
    context: Dict[str, Any] = field(default_factory=dict)
    provider: Optional[LLMProvider] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMResponse:
    """LLMレスポンス"""
    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, int]
    response_time: float
    cached: bool = False
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "content": self.content,
            "provider": self.provider.value,
            "model": self.model,
            "usage": self.usage,
            "response_time": self.response_time,
            "cached": self.cached,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class IncidentAnalysisResult:
    """インシデント分析結果"""
    root_cause: str
    mitigation_steps: List[str]
    prevention_strategies: List[str]
    related_patterns: List[str]
    severity_assessment: str
    estimated_impact: str
    confidence_score: float


@dataclass
class MetricAnalysisResult:
    """メトリクス分析結果"""
    health_status: str  # healthy, warning, critical
    trending_issues: List[str]
    performance_bottlenecks: List[str]
    recommended_actions: List[str]
    risk_level: str
    time_to_action: str


class ResponseCache:
    """レスポンスキャッシュ"""
    
    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 3600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[LLMResponse, datetime]] = {}
    
    def _generate_key(self, request: LLMRequest) -> str:
        """キャッシュキー生成"""
        content = f"{request.prompt}:{request.template}:{json.dumps(request.context, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """キャッシュからレスポンス取得"""
        key = self._generate_key(request)
        
        if key in self._cache:
            response, cached_at = self._cache[key]
            
            # TTL チェック
            if datetime.now() - cached_at < timedelta(seconds=self.ttl_seconds):
                response.cached = True
                logger.debug("Cache hit", cache_key=key[:8])
                return response
            else:
                # 期限切れエントリを削除
                del self._cache[key]
                logger.debug("Cache entry expired", cache_key=key[:8])
        
        return None
    
    def put(self, request: LLMRequest, response: LLMResponse):
        """レスポンスをキャッシュに保存"""
        key = self._generate_key(request)
        
        # キャッシュサイズ制限
        if len(self._cache) >= self.max_entries:
            # 最も古いエントリを削除
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        self._cache[key] = (response, datetime.now())
        logger.debug("Response cached", cache_key=key[:8])


class LLMWrapper:
    """LLM統合ラッパー"""
    
    def __init__(self):
        """初期化"""
        self.config_manager = get_config_manager()
        self.config = self.config_manager.get_module_config("llm_wrapper")
        
        # 設定値取得
        llm_config = self.config.config
        self.default_provider = LLMProvider(llm_config.get("default_provider", "claude"))
        self.timeout = llm_config.get("timeout", 60)
        self.max_retries = llm_config.get("max_retries", 3)
        self.retry_delay = llm_config.get("retry_delay", 1)
        
        # プロバイダー設定
        self.providers_config = llm_config.get("providers", {})
        self.system_prompts = llm_config.get("system_prompts", {})
        
        # キャッシュ設定
        cache_config = llm_config.get("cache", {})
        if cache_config.get("enabled", True):
            self.cache = ResponseCache(
                max_entries=cache_config.get("max_entries", 1000),
                ttl_seconds=cache_config.get("ttl_seconds", 3600)
            )
        else:
            self.cache = None
        
        # クライアント初期化
        self.clients: Dict[LLMProvider, Any] = {}
        self._init_clients()
        
        # 統計情報
        self.stats = {
            "requests_total": 0,
            "requests_cached": 0,
            "requests_by_provider": {},
            "errors_total": 0,
            "avg_response_time": 0.0
        }
        
        logger.info("LLM Wrapper initialized",
                   default_provider=self.default_provider.value,
                   enabled_providers=list(self.clients.keys()),
                   cache_enabled=bool(self.cache))
    
    def _init_clients(self):
        """LLMクライアント初期化"""
        # Claude クライアント
        claude_config = self.providers_config.get("claude", {})
        if claude_config.get("enabled", False):
            api_key = self._get_api_key("claude")
            if api_key:
                try:
                    config = ClaudeAPIConfig(
                        api_key=api_key,
                        model=claude_config.get("model", ClaudeModel.HAIKU.value),
                        max_tokens=claude_config.get("max_tokens", 4096),
                        temperature=claude_config.get("temperature", 0.1),
                        base_url=claude_config.get("base_url", "https://api.anthropic.com"),
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                        retry_delay=self.retry_delay
                    )
                    self.clients[LLMProvider.CLAUDE] = config
                    logger.info("Claude client configured", model=config.model)
                except Exception as e:
                    logger.error("Failed to configure Claude client", error=str(e))
        
        # OpenAI クライアント（将来の実装用）
        # openai_config = self.providers_config.get("openai", {})
        # if openai_config.get("enabled", False):
        #     # TODO: OpenAI クライアント実装
        #     pass
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """API キー取得"""
        env_var_map = {
            "claude": "AALS_CLAUDE_API_KEY",
            "openai": "AALS_OPENAI_API_KEY"
        }
        
        env_var = env_var_map.get(provider)
        if env_var:
            return os.getenv(env_var)
        return None
    
    def _get_system_prompt(self, template: PromptTemplate) -> str:
        """システムプロンプト取得"""
        return self.system_prompts.get(template.value, "")
    
    async def verify_setup(self) -> bool:
        """セットアップ確認"""
        if not self.clients:
            logger.error("No LLM clients configured")
            return False
        
        try:
            # デフォルトプロバイダーで接続確認
            if self.default_provider in self.clients:
                async with ClaudeAPIClient(self.clients[self.default_provider]) as client:
                    connected = await client.verify_connection()
                    
                    if connected:
                        logger.info("LLM Wrapper setup verification completed successfully")
                        return True
                    else:
                        logger.error("LLM Wrapper setup verification failed - connection failed")
                        return False
            else:
                logger.error("Default provider not configured", 
                           provider=self.default_provider.value)
                return False
                
        except Exception as e:
            logger.error("LLM Wrapper setup verification failed",
                        error=str(e), exception_type=type(e).__name__)
            return False
    
    async def generate_response(
        self,
        request: LLMRequest
    ) -> LLMResponse:
        """レスポンス生成"""
        start_time = time.time()
        
        # キャッシュ確認
        if self.cache:
            cached_response = self.cache.get(request)
            if cached_response:
                self.stats["requests_cached"] += 1
                return cached_response
        
        # プロバイダー決定
        provider = request.provider or self.default_provider
        
        if provider not in self.clients:
            raise ValueError(f"Provider {provider.value} not configured")
        
        # システムプロンプト取得
        system_prompt = None
        if request.template:
            system_prompt = self._get_system_prompt(request.template)
        
        try:
            # Claude API呼び出し
            if provider == LLMProvider.CLAUDE:
                response = await self._call_claude(request, system_prompt)
            else:
                raise NotImplementedError(f"Provider {provider.value} not implemented")
            
            # キャッシュ保存
            if self.cache:
                self.cache.put(request, response)
            
            # 統計更新
            self.stats["requests_total"] += 1
            self.stats["requests_by_provider"][provider.value] = \
                self.stats["requests_by_provider"].get(provider.value, 0) + 1
            
            response_time = time.time() - start_time
            self.stats["avg_response_time"] = \
                (self.stats["avg_response_time"] * (self.stats["requests_total"] - 1) + response_time) / \
                self.stats["requests_total"]
            
            logger.info("LLM response generated",
                       provider=provider.value,
                       template=request.template.value if request.template else None,
                       response_time=response_time,
                       content_length=len(response.content))
            
            return response
            
        except Exception as e:
            self.stats["errors_total"] += 1
            logger.error("LLM response generation failed",
                        provider=provider.value,
                        error=str(e),
                        exception_type=type(e).__name__)
            raise
    
    async def _call_claude(self, request: LLMRequest, system_prompt: Optional[str]) -> LLMResponse:
        """Claude API呼び出し"""
        config = self.clients[LLMProvider.CLAUDE]
        
        async with ClaudeAPIClient(config) as client:
            claude_response = await client.send_single_message(
                user_message=request.prompt,
                system_prompt=system_prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            return LLMResponse(
                content=claude_response.content,
                provider=LLMProvider.CLAUDE,
                model=claude_response.model,
                usage=claude_response.usage,
                response_time=claude_response.response_time,
                request_id=claude_response.request_id
            )
    
    async def analyze_incident(
        self,
        incident_description: str,
        metrics_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> IncidentAnalysisResult:
        """インシデント分析"""
        prompt = f"""
        Incident Description:
        {incident_description}
        
        """
        
        if metrics_data:
            prompt += f"Metrics Data:\n{json.dumps(metrics_data, indent=2)}\n\n"
        
        if context:
            prompt += f"Additional Context:\n{json.dumps(context, indent=2)}\n\n"
        
        prompt += """
        Please provide a comprehensive incident analysis with the following structure:
        
        ROOT_CAUSE: [Detailed root cause analysis]
        
        MITIGATION_STEPS:
        1. [Immediate action 1]
        2. [Immediate action 2]
        3. [etc.]
        
        PREVENTION_STRATEGIES:
        1. [Prevention strategy 1]
        2. [Prevention strategy 2]
        3. [etc.]
        
        RELATED_PATTERNS:
        - [Pattern 1]
        - [Pattern 2]
        - [etc.]
        
        SEVERITY: [CRITICAL/HIGH/MEDIUM/LOW]
        IMPACT: [Description of estimated impact]
        CONFIDENCE: [0.0-1.0]
        """
        
        request = LLMRequest(
            prompt=prompt,
            template=PromptTemplate.INCIDENT_ANALYSIS,
            context=context or {}
        )
        
        response = await self.generate_response(request)
        
        # レスポンス解析
        return self._parse_incident_analysis(response.content)
    
    async def analyze_metrics(
        self,
        metrics_data: Dict[str, Any],
        time_range: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MetricAnalysisResult:
        """メトリクス分析"""
        prompt = f"""
        Metrics Data:
        {json.dumps(metrics_data, indent=2)}
        
        """
        
        if time_range:
            prompt += f"Time Range: {time_range}\n\n"
        
        if context:
            prompt += f"Context:\n{json.dumps(context, indent=2)}\n\n"
        
        prompt += """
        Please analyze the metrics and provide assessment with this structure:
        
        HEALTH_STATUS: [healthy/warning/critical]
        
        TRENDING_ISSUES:
        - [Issue 1]
        - [Issue 2]
        - [etc.]
        
        BOTTLENECKS:
        - [Bottleneck 1]
        - [Bottleneck 2]
        - [etc.]
        
        ACTIONS:
        1. [Recommended action 1]
        2. [Recommended action 2]
        3. [etc.]
        
        RISK_LEVEL: [LOW/MEDIUM/HIGH/CRITICAL]
        TIME_TO_ACTION: [immediate/within_hour/within_day/monitoring]
        """
        
        request = LLMRequest(
            prompt=prompt,
            template=PromptTemplate.METRIC_ANALYSIS,
            context=context or {}
        )
        
        response = await self.generate_response(request)
        
        return self._parse_metric_analysis(response.content)
    
    async def generate_solution(
        self,
        problem_description: str,
        error_logs: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """解決策生成"""
        prompt = f"""
        Problem Description:
        {problem_description}
        
        """
        
        if error_logs:
            prompt += "Error Logs:\n"
            for i, log in enumerate(error_logs[:5], 1):  # 最大5件
                prompt += f"{i}. {log}\n"
            prompt += "\n"
        
        if context:
            prompt += f"Context:\n{json.dumps(context, indent=2)}\n\n"
        
        prompt += """
        Please provide step-by-step troubleshooting solutions:
        
        SOLUTIONS:
        1. [Step-by-step solution 1]
        2. [Step-by-step solution 2]
        3. [etc.]
        
        Each solution should be actionable and specific.
        """
        
        request = LLMRequest(
            prompt=prompt,
            template=PromptTemplate.SOLUTION_GENERATION,
            context=context or {}
        )
        
        response = await self.generate_response(request)
        
        return self._parse_solutions(response.content)
    
    def _parse_incident_analysis(self, content: str) -> IncidentAnalysisResult:
        """インシデント分析結果をパース"""
        # 簡単なパーサー実装（実際にはより堅牢な実装が必要）
        lines = content.split('\n')
        
        root_cause = ""
        mitigation_steps = []
        prevention_strategies = []
        related_patterns = []
        severity = "MEDIUM"
        impact = ""
        confidence = 0.5
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("ROOT_CAUSE:"):
                root_cause = line.replace("ROOT_CAUSE:", "").strip()
                current_section = None
            elif line.startswith("MITIGATION_STEPS:"):
                current_section = "mitigation"
            elif line.startswith("PREVENTION_STRATEGIES:"):
                current_section = "prevention"
            elif line.startswith("RELATED_PATTERNS:"):
                current_section = "patterns"
            elif line.startswith("SEVERITY:"):
                severity = line.replace("SEVERITY:", "").strip()
                current_section = None
            elif line.startswith("IMPACT:"):
                impact = line.replace("IMPACT:", "").strip()
                current_section = None
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.5
                current_section = None
            elif line and current_section:
                if current_section == "mitigation" and (line.startswith(tuple('123456789')) or line.startswith('-')):
                    mitigation_steps.append(line.lstrip('0123456789.- '))
                elif current_section == "prevention" and (line.startswith(tuple('123456789')) or line.startswith('-')):
                    prevention_strategies.append(line.lstrip('0123456789.- '))
                elif current_section == "patterns" and line.startswith('-'):
                    related_patterns.append(line.lstrip('- '))
        
        return IncidentAnalysisResult(
            root_cause=root_cause,
            mitigation_steps=mitigation_steps,
            prevention_strategies=prevention_strategies,
            related_patterns=related_patterns,
            severity_assessment=severity,
            estimated_impact=impact,
            confidence_score=confidence
        )
    
    def _parse_metric_analysis(self, content: str) -> MetricAnalysisResult:
        """メトリクス分析結果をパース"""
        lines = content.split('\n')
        
        health_status = "warning"
        trending_issues = []
        bottlenecks = []
        actions = []
        risk_level = "MEDIUM"
        time_to_action = "monitoring"
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("HEALTH_STATUS:"):
                health_status = line.replace("HEALTH_STATUS:", "").strip().lower()
                current_section = None
            elif line.startswith("TRENDING_ISSUES:"):
                current_section = "issues"
            elif line.startswith("BOTTLENECKS:"):
                current_section = "bottlenecks"
            elif line.startswith("ACTIONS:"):
                current_section = "actions"
            elif line.startswith("RISK_LEVEL:"):
                risk_level = line.replace("RISK_LEVEL:", "").strip()
                current_section = None
            elif line.startswith("TIME_TO_ACTION:"):
                time_to_action = line.replace("TIME_TO_ACTION:", "").strip()
                current_section = None
            elif line and current_section:
                if current_section == "issues" and line.startswith('-'):
                    trending_issues.append(line.lstrip('- '))
                elif current_section == "bottlenecks" and line.startswith('-'):
                    bottlenecks.append(line.lstrip('- '))
                elif current_section == "actions" and (line.startswith(tuple('123456789')) or line.startswith('-')):
                    actions.append(line.lstrip('0123456789.- '))
        
        return MetricAnalysisResult(
            health_status=health_status,
            trending_issues=trending_issues,
            performance_bottlenecks=bottlenecks,
            recommended_actions=actions,
            risk_level=risk_level,
            time_to_action=time_to_action
        )
    
    def _parse_solutions(self, content: str) -> List[str]:
        """解決策をパース"""
        lines = content.split('\n')
        solutions = []
        in_solutions = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("SOLUTIONS:"):
                in_solutions = True
                continue
            
            if in_solutions and line and (line.startswith(tuple('123456789')) or line.startswith('-')):
                solutions.append(line.lstrip('0123456789.- '))
        
        return solutions
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        cache_stats = {}
        if self.cache:
            cache_stats = {
                "cache_size": len(self.cache._cache),
                "cache_hit_rate": self.stats["requests_cached"] / max(1, self.stats["requests_total"])
            }
        
        return {
            **self.stats,
            **cache_stats,
            "configured_providers": list(self.clients.keys()),
            "default_provider": self.default_provider.value
        }
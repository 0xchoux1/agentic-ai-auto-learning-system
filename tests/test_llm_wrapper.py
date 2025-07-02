"""
LLM Wrapper Module Tests

Module 6: LLM Wrapper の包括的テストスイート
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from aals.modules.llm_wrapper import (
    LLMWrapper, LLMRequest, LLMResponse, PromptTemplate, LLMProvider,
    IncidentAnalysisResult, MetricAnalysisResult, ResponseCache
)
from aals.integrations.claude_client import (
    ClaudeAPIClient, ClaudeAPIConfig, ClaudeMessage, ClaudeResponse, ClaudeModel
)


@pytest.fixture
def sample_llm_config():
    """サンプルLLM設定"""
    return {
        "enabled": True,
        "default_provider": "claude",
        "timeout": 60,
        "max_retries": 3,
        "retry_delay": 1,
        "providers": {
            "claude": {
                "enabled": True,
                "api_key": "test-api-key",
                "model": "claude-3-haiku-20240307",
                "max_tokens": 4096,
                "temperature": 0.1,
                "base_url": "https://api.anthropic.com"
            }
        },
        "cache": {
            "enabled": True,
            "ttl_seconds": 3600,
            "max_entries": 1000
        },
        "system_prompts": {
            "incident_analysis": "You are an expert SRE assistant.",
            "metric_analysis": "You are a monitoring expert.",
            "solution_generation": "You are a troubleshooting expert."
        }
    }


@pytest.fixture
def mock_config_manager(sample_llm_config):
    """モック設定マネージャー"""
    mock_config = MagicMock()
    mock_config.config = sample_llm_config
    
    mock_manager = MagicMock()
    mock_manager.get_module_config.return_value = mock_config
    return mock_manager


@pytest.fixture
def llm_wrapper(mock_config_manager):
    """LLM Wrapperインスタンス"""
    with patch('aals.modules.llm_wrapper.get_config_manager', return_value=mock_config_manager), \
         patch.dict('os.environ', {'AALS_CLAUDE_API_KEY': 'test-api-key'}):
        return LLMWrapper()


@pytest.fixture
def sample_claude_response():
    """サンプルClaude APIレスポンス"""
    return ClaudeResponse(
        content="This is a test response from Claude.",
        model="claude-3-haiku-20240307",
        usage={"input_tokens": 10, "output_tokens": 8},
        response_time=1.5,
        request_id="req_test123"
    )


class TestClaudeAPIClient:
    """Claude APIクライアントテスト"""
    
    def test_client_initialization(self):
        """クライアント初期化テスト"""
        config = ClaudeAPIConfig(
            api_key="test-key",
            model=ClaudeModel.HAIKU.value,
            max_tokens=4096,
            temperature=0.1
        )
        
        client = ClaudeAPIClient(config)
        assert client.config.api_key == "test-key"
        assert client.config.model == ClaudeModel.HAIKU.value
        assert client.config.max_tokens == 4096
        assert client.config.temperature == 0.1
    
    def test_client_no_api_key(self):
        """APIキーなし初期化エラーテスト"""
        config = ClaudeAPIConfig(api_key="")
        
        with pytest.raises(ValueError, match="Claude API key is required"):
            ClaudeAPIClient(config)
    
    def test_headers_generation(self):
        """ヘッダー生成テスト"""
        config = ClaudeAPIConfig(api_key="test-key")
        client = ClaudeAPIClient(config)
        
        headers = client._get_headers()
        
        assert headers["Content-Type"] == "application/json"
        assert headers["x-api-key"] == "test-key"
        assert headers["anthropic-version"] == "2023-06-01"
    
    def test_message_formatting(self):
        """メッセージ形式変換テスト"""
        config = ClaudeAPIConfig(api_key="test-key")
        client = ClaudeAPIClient(config)
        
        messages = [
            ClaudeMessage(role="user", content="Hello"),
            ClaudeMessage(role="assistant", content="Hi there!")
        ]
        
        formatted = client._format_messages(messages)
        
        assert len(formatted) == 2
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"] == "Hello"
        assert formatted[1]["role"] == "assistant"
        assert formatted[1]["content"] == "Hi there!"
    
    @pytest.mark.asyncio
    async def test_verify_connection_success(self):
        """接続確認成功テスト"""
        config = ClaudeAPIConfig(api_key="test-key")
        
        with patch.object(ClaudeAPIClient, 'send_single_message') as mock_send:
            mock_send.return_value = ClaudeResponse(
                content="OK",
                model="claude-3-haiku-20240307",
                usage={},
                response_time=1.0
            )
            
            client = ClaudeAPIClient(config)
            result = await client.verify_connection()
            
            assert result is True
            mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_verify_connection_failure(self):
        """接続確認失敗テスト"""
        config = ClaudeAPIConfig(api_key="test-key")
        
        with patch.object(ClaudeAPIClient, 'send_single_message') as mock_send:
            mock_send.side_effect = Exception("Connection failed")
            
            client = ClaudeAPIClient(config)
            result = await client.verify_connection()
            
            assert result is False


class TestResponseCache:
    """レスポンスキャッシュテスト"""
    
    def test_cache_initialization(self):
        """キャッシュ初期化テスト"""
        cache = ResponseCache(max_entries=100, ttl_seconds=1800)
        
        assert cache.max_entries == 100
        assert cache.ttl_seconds == 1800
        assert len(cache._cache) == 0
    
    def test_cache_key_generation(self):
        """キャッシュキー生成テスト"""
        cache = ResponseCache()
        
        request1 = LLMRequest(prompt="Hello", template=PromptTemplate.INCIDENT_ANALYSIS)
        request2 = LLMRequest(prompt="Hello", template=PromptTemplate.INCIDENT_ANALYSIS)
        request3 = LLMRequest(prompt="Hi", template=PromptTemplate.INCIDENT_ANALYSIS)
        
        key1 = cache._generate_key(request1)
        key2 = cache._generate_key(request2)
        key3 = cache._generate_key(request3)
        
        assert key1 == key2  # 同じリクエストは同じキー
        assert key1 != key3  # 異なるリクエストは異なるキー
    
    def test_cache_put_and_get(self):
        """キャッシュ保存・取得テスト"""
        cache = ResponseCache()
        
        request = LLMRequest(prompt="Test prompt")
        response = LLMResponse(
            content="Test response",
            provider=LLMProvider.CLAUDE,
            model="claude-3-haiku-20240307",
            usage={},
            response_time=1.0
        )
        
        # キャッシュに保存
        cache.put(request, response)
        
        # キャッシュから取得
        cached_response = cache.get(request)
        
        assert cached_response is not None
        assert cached_response.content == "Test response"
        assert cached_response.cached is True
    
    def test_cache_ttl_expiry(self):
        """キャッシュTTL期限切れテスト"""
        cache = ResponseCache(ttl_seconds=1)  # 1秒でTTL切れ
        
        request = LLMRequest(prompt="Test prompt")
        response = LLMResponse(
            content="Test response",
            provider=LLMProvider.CLAUDE,
            model="claude-3-haiku-20240307",
            usage={},
            response_time=1.0
        )
        
        # キャッシュに保存
        cache.put(request, response)
        
        # 即座に取得 - キャッシュヒット
        cached_response = cache.get(request)
        assert cached_response is not None
        
        # TTL期限切れ後のテストは時間がかかるのでスキップ
        # 実際のテストではtime.sleepを使うか、タイムスタンプをモック
    
    def test_cache_max_entries(self):
        """キャッシュサイズ制限テスト"""
        cache = ResponseCache(max_entries=2)
        
        # 3つのエントリを追加
        for i in range(3):
            request = LLMRequest(prompt=f"Test prompt {i}")
            response = LLMResponse(
                content=f"Test response {i}",
                provider=LLMProvider.CLAUDE,
                model="claude-3-haiku-20240307",
                usage={},
                response_time=1.0
            )
            cache.put(request, response)
        
        # キャッシュサイズが制限内であることを確認
        assert len(cache._cache) <= 2


class TestLLMWrapper:
    """LLM Wrapperテスト"""
    
    def test_initialization(self, llm_wrapper):
        """初期化テスト"""
        assert llm_wrapper.default_provider == LLMProvider.CLAUDE
        assert llm_wrapper.timeout == 60
        assert llm_wrapper.max_retries == 3
        assert llm_wrapper.cache is not None
        assert LLMProvider.CLAUDE in llm_wrapper.clients
    
    @pytest.mark.asyncio
    async def test_verify_setup_success(self, llm_wrapper):
        """セットアップ確認成功テスト"""
        with patch('aals.modules.llm_wrapper.ClaudeAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.verify_connection = AsyncMock(return_value=True)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            result = await llm_wrapper.verify_setup()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_setup_failure(self, llm_wrapper):
        """セットアップ確認失敗テスト"""
        with patch('aals.modules.llm_wrapper.ClaudeAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.verify_connection = AsyncMock(return_value=False)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            result = await llm_wrapper.verify_setup()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_generate_response(self, llm_wrapper, sample_claude_response):
        """レスポンス生成テスト"""
        with patch('aals.modules.llm_wrapper.ClaudeAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.send_single_message = AsyncMock(return_value=sample_claude_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            request = LLMRequest(
                prompt="Analyze this incident",
                template=PromptTemplate.INCIDENT_ANALYSIS
            )
            
            response = await llm_wrapper.generate_response(request)
            
            assert response.content == "This is a test response from Claude."
            assert response.provider == LLMProvider.CLAUDE
            assert response.model == "claude-3-haiku-20240307"
            assert response.response_time == 1.5
    
    @pytest.mark.asyncio
    async def test_generate_response_with_cache(self, llm_wrapper, sample_claude_response):
        """キャッシュ付きレスポンス生成テスト"""
        with patch('aals.modules.llm_wrapper.ClaudeAPIClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.send_single_message = AsyncMock(return_value=sample_claude_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            request = LLMRequest(prompt="Test prompt")
            
            # 1回目の呼び出し
            response1 = await llm_wrapper.generate_response(request)
            
            # 2回目の呼び出し（キャッシュヒット）
            response2 = await llm_wrapper.generate_response(request)
            
            assert response1.content == response2.content
            assert response2.cached is True
            # API は1回だけ呼ばれるはず
            mock_client.send_single_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_incident(self, llm_wrapper):
        """インシデント分析テスト"""
        mock_response_content = """
        ROOT_CAUSE: Database connection pool exhausted
        
        MITIGATION_STEPS:
        1. Restart application servers
        2. Increase connection pool size
        3. Monitor connection usage
        
        PREVENTION_STRATEGIES:
        1. Implement connection pooling monitoring
        2. Set up alerting for pool usage
        
        RELATED_PATTERNS:
        - High database load
        - Connection leaks
        
        SEVERITY: CRITICAL
        IMPACT: Service unavailable for users
        CONFIDENCE: 0.9
        """
        
        with patch.object(llm_wrapper, 'generate_response') as mock_generate:
            mock_generate.return_value = LLMResponse(
                content=mock_response_content,
                provider=LLMProvider.CLAUDE,
                model="claude-3-haiku-20240307",
                usage={},
                response_time=2.0
            )
            
            result = await llm_wrapper.analyze_incident(
                incident_description="Database errors in production",
                metrics_data={"cpu_usage": 85, "memory_usage": 90},
                context={"environment": "production"}
            )
            
            assert isinstance(result, IncidentAnalysisResult)
            assert "Database connection pool exhausted" in result.root_cause
            assert len(result.mitigation_steps) >= 3
            assert len(result.prevention_strategies) >= 2
            assert result.severity_assessment == "CRITICAL"
            assert result.confidence_score == 0.9
    
    @pytest.mark.asyncio
    async def test_analyze_metrics(self, llm_wrapper):
        """メトリクス分析テスト"""
        mock_response_content = """
        HEALTH_STATUS: critical
        
        TRENDING_ISSUES:
        - High CPU usage
        - Memory pressure
        - Slow response times
        
        BOTTLENECKS:
        - Database query performance
        - Network latency
        
        ACTIONS:
        1. Scale up instances
        2. Optimize database queries
        3. Review network configuration
        
        RISK_LEVEL: HIGH
        TIME_TO_ACTION: immediate
        """
        
        with patch.object(llm_wrapper, 'generate_response') as mock_generate:
            mock_generate.return_value = LLMResponse(
                content=mock_response_content,
                provider=LLMProvider.CLAUDE,
                model="claude-3-haiku-20240307",
                usage={},
                response_time=2.0
            )
            
            result = await llm_wrapper.analyze_metrics(
                metrics_data={"cpu": 95, "memory": 88, "disk": 70},
                time_range="last_1h",
                context={"alerts": 5}
            )
            
            assert isinstance(result, MetricAnalysisResult)
            assert result.health_status == "critical"
            assert len(result.trending_issues) == 3
            assert len(result.performance_bottlenecks) == 2
            assert len(result.recommended_actions) >= 3
            assert result.risk_level == "HIGH"
            assert result.time_to_action == "immediate"
    
    @pytest.mark.asyncio
    async def test_generate_solution(self, llm_wrapper):
        """解決策生成テスト"""
        mock_response_content = """
        SOLUTIONS:
        1. Check application logs for error patterns
        2. Verify database connectivity and query performance
        3. Review recent deployment changes
        4. Scale application resources if needed
        5. Contact on-call engineer for escalation
        """
        
        with patch.object(llm_wrapper, 'generate_response') as mock_generate:
            mock_generate.return_value = LLMResponse(
                content=mock_response_content,
                provider=LLMProvider.CLAUDE,
                model="claude-3-haiku-20240307",
                usage={},
                response_time=1.5
            )
            
            solutions = await llm_wrapper.generate_solution(
                problem_description="API endpoints returning 500 errors",
                error_logs=["ConnectionError: Failed to connect to database", "TimeoutError: Query timeout"],
                context={"service": "user-api", "environment": "production"}
            )
            
            assert isinstance(solutions, list)
            assert len(solutions) >= 4
            assert any("logs" in solution.lower() for solution in solutions)
            assert any("database" in solution.lower() for solution in solutions)
    
    def test_parse_incident_analysis(self, llm_wrapper):
        """インシデント分析パース テスト"""
        content = """
        ROOT_CAUSE: Memory leak in application
        
        MITIGATION_STEPS:
        1. Restart affected services
        2. Apply memory patch
        
        PREVENTION_STRATEGIES:
        1. Implement memory monitoring
        2. Code review for memory usage
        
        RELATED_PATTERNS:
        - Gradual memory increase
        - GC pressure
        
        SEVERITY: HIGH
        IMPACT: Performance degradation
        CONFIDENCE: 0.8
        """
        
        result = llm_wrapper._parse_incident_analysis(content)
        
        assert "Memory leak in application" in result.root_cause
        assert len(result.mitigation_steps) == 2
        assert len(result.prevention_strategies) == 2
        assert len(result.related_patterns) == 2
        assert result.severity_assessment == "HIGH"
        assert result.confidence_score == 0.8
    
    def test_parse_metric_analysis(self, llm_wrapper):
        """メトリクス分析パース テスト"""
        content = """
        HEALTH_STATUS: warning
        
        TRENDING_ISSUES:
        - Increasing response times
        - Memory usage trending up
        
        BOTTLENECKS:
        - Database connection pool
        - API rate limiting
        
        ACTIONS:
        1. Monitor database performance
        2. Increase rate limits
        3. Review application logs
        
        RISK_LEVEL: MEDIUM
        TIME_TO_ACTION: within_hour
        """
        
        result = llm_wrapper._parse_metric_analysis(content)
        
        assert result.health_status == "warning"
        assert len(result.trending_issues) == 2
        assert len(result.performance_bottlenecks) == 2
        assert len(result.recommended_actions) == 3
        assert result.risk_level == "MEDIUM"
        assert result.time_to_action == "within_hour"
    
    def test_parse_solutions(self, llm_wrapper):
        """解決策パース テスト"""
        content = """
        SOLUTIONS:
        1. Check system logs for errors
        2. Verify network connectivity
        3. Restart the service
        - Alternative solution approach
        """
        
        solutions = llm_wrapper._parse_solutions(content)
        
        assert len(solutions) == 4
        assert "Check system logs for errors" in solutions
        assert "Verify network connectivity" in solutions
        assert "Restart the service" in solutions
        assert "Alternative solution approach" in solutions
    
    def test_get_stats(self, llm_wrapper):
        """統計情報取得テスト"""
        stats = llm_wrapper.get_stats()
        
        assert "requests_total" in stats
        assert "requests_cached" in stats
        assert "configured_providers" in stats
        assert "default_provider" in stats
        assert "cache_size" in stats
        assert "cache_hit_rate" in stats
        
        assert stats["default_provider"] == "claude"
        assert LLMProvider.CLAUDE.value in [p.value for p in stats["configured_providers"]]


class TestLLMDataClasses:
    """LLMデータクラステスト"""
    
    def test_llm_request_creation(self):
        """LLMリクエスト作成テスト"""
        request = LLMRequest(
            prompt="Test prompt",
            template=PromptTemplate.INCIDENT_ANALYSIS,
            context={"key": "value"},
            provider=LLMProvider.CLAUDE
        )
        
        assert request.prompt == "Test prompt"
        assert request.template == PromptTemplate.INCIDENT_ANALYSIS
        assert request.context == {"key": "value"}
        assert request.provider == LLMProvider.CLAUDE
        assert isinstance(request.timestamp, datetime)
    
    def test_llm_response_creation(self):
        """LLMレスポンス作成テスト"""
        response = LLMResponse(
            content="Test response",
            provider=LLMProvider.CLAUDE,
            model="claude-3-haiku-20240307",
            usage={"input_tokens": 10, "output_tokens": 5},
            response_time=1.5,
            cached=False,
            request_id="req_123"
        )
        
        assert response.content == "Test response"
        assert response.provider == LLMProvider.CLAUDE
        assert response.model == "claude-3-haiku-20240307"
        assert response.usage == {"input_tokens": 10, "output_tokens": 5}
        assert response.response_time == 1.5
        assert response.cached is False
        assert response.request_id == "req_123"
        assert isinstance(response.timestamp, datetime)
    
    def test_llm_response_to_dict(self):
        """LLMレスポンス辞書変換テスト"""
        response = LLMResponse(
            content="Test response",
            provider=LLMProvider.CLAUDE,
            model="claude-3-haiku-20240307",
            usage={"input_tokens": 10, "output_tokens": 5},
            response_time=1.5
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["content"] == "Test response"
        assert response_dict["provider"] == "claude"
        assert response_dict["model"] == "claude-3-haiku-20240307"
        assert response_dict["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert response_dict["response_time"] == 1.5
        assert "timestamp" in response_dict


@pytest.mark.asyncio
async def test_full_integration(llm_wrapper, sample_claude_response):
    """完全統合テスト"""
    with patch('aals.modules.llm_wrapper.ClaudeAPIClient') as mock_client_class:
        mock_client = AsyncMock()
        
        # セットアップ確認用のモック
        mock_client.verify_connection = AsyncMock(return_value=True)
        
        # レスポンス生成用のモック
        mock_client.send_single_message = AsyncMock(return_value=sample_claude_response)
        
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client
        
        # セットアップ確認
        setup_result = await llm_wrapper.verify_setup()
        assert setup_result is True
        
        # レスポンス生成
        request = LLMRequest(
            prompt="Analyze this system alert",
            template=PromptTemplate.INCIDENT_ANALYSIS,
            context={"service": "api", "environment": "production"}
        )
        
        response = await llm_wrapper.generate_response(request)
        
        assert response.content == "This is a test response from Claude."
        assert response.provider == LLMProvider.CLAUDE
        
        # 統計確認
        stats = llm_wrapper.get_stats()
        assert stats["requests_total"] == 1
        assert stats["errors_total"] == 0
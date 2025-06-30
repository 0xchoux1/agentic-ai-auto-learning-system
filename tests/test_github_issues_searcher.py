#!/usr/bin/env python3
"""
AALS Module 5: GitHub Issues Searcher テスト
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from aals.modules.github_issues_searcher import (
    GitHubIssuesSearcher,
    SimilarIssue,
    IncidentPattern,
    GitHubSearchReport
)
from aals.integrations.github_client import (
    GitHubAPIClient,
    GitHubIssue,
    SearchQuery,
    IssueState,
    IssuePriority
)


@pytest.fixture  
def sample_github_issues():
    """サンプルGitHub Issues"""
    now = datetime.now()
    return [
        GitHubIssue(
            number=123,
            title="Server down - urgent fix needed",
            body="Production server is not responding. Need immediate fix.",
            state="closed",
            labels=["critical", "bug", "production"],
            assignees=["john_doe"],
            created_at=now - timedelta(days=5),
            updated_at=now - timedelta(days=4),
            closed_at=now - timedelta(days=4),
            author="alice",
            comments_count=8,
            url="https://github.com/owner/repo/issues/123",
            repository="owner/repo"
        ),
        GitHubIssue(
            number=124,
            title="Database connection timeout",
            body="Database queries are timing out frequently.",
            state="open",
            labels=["bug", "database", "performance"],
            assignees=["jane_doe"],
            created_at=now - timedelta(days=2),
            updated_at=now - timedelta(hours=6),
            closed_at=None,
            author="bob",
            comments_count=3,
            url="https://github.com/owner/repo/issues/124",
            repository="owner/repo"
        ),
        GitHubIssue(
            number=125,
            title="API response slow",
            body="API endpoints are responding slowly during peak hours.",
            state="closed",
            labels=["performance", "api"],
            assignees=["john_doe"],
            created_at=now - timedelta(days=10),
            updated_at=now - timedelta(days=8),
            closed_at=now - timedelta(days=8),
            author="charlie",
            comments_count=12,
            url="https://github.com/owner/repo/issues/125",
            repository="owner/repo"
        )
    ]


@pytest.fixture
def github_issues_searcher():
    """GitHub Issues Searcher インスタンス"""
    with patch('aals.modules.github_issues_searcher.get_config_manager') as mock_config:
        mock_module_config = MagicMock()
        mock_module_config.enabled = True
        mock_module_config.config = {
            "token": "test_token",
            "repositories": ["owner/repo1", "owner/repo2"],
            "timeout": 30,
            "max_results": 50,
            "similarity_threshold": 0.3
        }
        mock_config.return_value.get_module_config.return_value = mock_module_config
        
        searcher = GitHubIssuesSearcher()
        return searcher


class TestGitHubIssuesSearcher:
    """GitHubIssuesSearcher テスト"""
    
    def test_initialization(self, github_issues_searcher):
        """初期化テスト"""
        assert github_issues_searcher.github_token == "test_token"
        assert len(github_issues_searcher.default_repositories) == 2
        assert github_issues_searcher.max_results == 50
        assert github_issues_searcher.similarity_threshold == 0.3
        assert isinstance(github_issues_searcher.github_client, GitHubAPIClient)
    
    @pytest.mark.asyncio
    async def test_verify_setup_success(self, github_issues_searcher):
        """セットアップ確認成功テスト"""
        with patch.object(github_issues_searcher.github_client, 'verify_connection', return_value=True):
            with patch.object(github_issues_searcher.github_client, 'get_rate_limit_info', return_value={
                "search": {"remaining": 30}
            }):
                result = await github_issues_searcher.verify_setup()
                assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_setup_failure(self, github_issues_searcher):
        """セットアップ確認失敗テスト"""
        with patch.object(github_issues_searcher.github_client, 'verify_connection', return_value=False):
            result = await github_issues_searcher.verify_setup()
            assert result is False
    
    def test_calculate_text_similarity(self, github_issues_searcher):
        """テキスト類似度計算テスト"""
        text1 = "server down production critical fix"
        text2 = "production server critical issue needs fix"
        
        similarity = github_issues_searcher._calculate_text_similarity(text1, text2)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # 高い類似度が期待される
        
        # 全く異なるテキスト
        text3 = "documentation update readme file"
        similarity2 = github_issues_searcher._calculate_text_similarity(text1, text3)
        assert similarity2 < similarity  # より低い類似度
    
    def test_find_matching_keywords(self, github_issues_searcher, sample_github_issues):
        """マッチングキーワード検索テスト"""
        target_text = "server down production critical"
        issue = sample_github_issues[0]  # "Server down - urgent fix needed"
        
        keywords = github_issues_searcher._find_matching_keywords(target_text, issue)
        assert "server" in keywords
        assert "production" in keywords or "down" in keywords
        assert len(keywords) > 0
    
    def test_analyze_incident_patterns(self, github_issues_searcher, sample_github_issues):
        """インシデントパターン分析テスト"""
        patterns = github_issues_searcher._analyze_incident_patterns(sample_github_issues)
        
        assert len(patterns) > 0
        
        # パターンの構造確認
        for pattern in patterns:
            assert isinstance(pattern, IncidentPattern)
            assert pattern.frequency > 0
            assert isinstance(pattern.common_labels, list)
            assert isinstance(pattern.frequent_assignees, list)
    
    def test_generate_solutions(self, github_issues_searcher, sample_github_issues):
        """解決策生成テスト"""
        # Similar issues作成
        similar_issues = [
            SimilarIssue(
                issue=sample_github_issues[0],  # closed issue
                similarity_score=0.8,
                matching_keywords=["server", "production"],
                relevance_reason="High similarity"
            ),
            SimilarIssue(
                issue=sample_github_issues[1],  # open issue
                similarity_score=0.6,
                matching_keywords=["database"],
                relevance_reason="Medium similarity"
            )
        ]
        
        patterns = [
            IncidentPattern(
                incident_type="outage",
                frequency=2,
                avg_resolution_time=12.0,
                common_labels=["critical", "production"],
                frequent_assignees=["john_doe"],
                success_patterns=["Label: critical, production"]
            )
        ]
        
        solutions = github_issues_searcher._generate_solutions(similar_issues, patterns)
        
        assert len(solutions) > 0
        assert any("Similar resolved issues found" in sol for sol in solutions)
        assert any("john_doe" in sol for sol in solutions)
    
    def test_recommend_assignees(self, github_issues_searcher):
        """担当者推奨テスト"""
        patterns = [
            IncidentPattern(
                incident_type="performance",
                frequency=3,
                avg_resolution_time=8.0,
                common_labels=["performance", "api"],
                frequent_assignees=["john_doe", "jane_doe"],
                success_patterns=[]
            )
        ]
        
        issue_labels = ["performance", "bug"]
        assignees = github_issues_searcher._recommend_assignees(patterns, issue_labels)
        
        assert "john_doe" in assignees or "jane_doe" in assignees
        assert len(assignees) <= 3
    
    def test_estimate_resolution_time(self, github_issues_searcher):
        """解決時間予測テスト"""
        patterns = [
            IncidentPattern(
                incident_type="bug",
                frequency=2,
                avg_resolution_time=16.0,  # 16時間
                common_labels=[],
                frequent_assignees=[],
                success_patterns=[]
            )
        ]
        
        # Critical priority
        time_critical = github_issues_searcher._estimate_resolution_time(patterns, IssuePriority.CRITICAL)
        assert time_critical is not None
        assert time_critical < 24.0  # Critical should be resolved quickly
        
        # Low priority
        time_low = github_issues_searcher._estimate_resolution_time(patterns, IssuePriority.LOW)
        assert time_low is not None
        assert time_low > time_critical  # Low priority takes longer
    
    def test_recommend_priority(self, github_issues_searcher):
        """優先度推奨テスト"""
        # Critical keywords
        critical_keywords = ["critical", "production", "down"]
        priority = github_issues_searcher._recommend_priority([], critical_keywords)
        assert priority == IssuePriority.CRITICAL
        
        # High keywords
        high_keywords = ["urgent", "important"]
        priority = github_issues_searcher._recommend_priority([], high_keywords)
        assert priority == IssuePriority.HIGH
        
        # Normal keywords
        normal_keywords = ["documentation", "update"]
        priority = github_issues_searcher._recommend_priority([], normal_keywords)
        assert priority == IssuePriority.MEDIUM
    
    @pytest.mark.asyncio
    async def test_search_similar_issues(self, github_issues_searcher, sample_github_issues):
        """類似Issue検索テスト"""
        with patch.object(github_issues_searcher.github_client, 'search_issues', return_value=sample_github_issues):
            description = "Server performance issue"
            keywords = ["server", "performance", "slow"]
            
            report = await github_issues_searcher.search_similar_issues(
                description=description,
                keywords=keywords,
                repositories=["owner/repo"]
            )
            
            assert isinstance(report, GitHubSearchReport)
            assert report.query_description == description
            assert report.total_issues_found == len(sample_github_issues)
            assert len(report.similar_issues) >= 0
            assert len(report.suggested_solutions) > 0
            assert report.priority_recommendation in IssuePriority
            assert report.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_search_similar_issues_with_filters(self, github_issues_searcher, sample_github_issues):
        """フィルター付き類似Issue検索テスト"""
        # 過去24時間のみ
        recent_issue = sample_github_issues[1]  # 2日前のIssue
        recent_issue.created_at = datetime.now() - timedelta(hours=12)
        
        with patch.object(github_issues_searcher.github_client, 'search_issues', return_value=[recent_issue]):
            report = await github_issues_searcher.search_similar_issues(
                description="Recent database issue",
                keywords=["database", "timeout"],
                repositories=["owner/repo"],  # 単一リポジトリを指定
                hours_back=24,
                include_closed=False
            )
            
            assert report.total_issues_found == 1
    
    @pytest.mark.asyncio
    async def test_get_repository_insights(self, github_issues_searcher, sample_github_issues):
        """リポジトリインサイト取得テスト"""
        with patch.object(github_issues_searcher.github_client, 'get_repository_issues', return_value=sample_github_issues):
            insights = await github_issues_searcher.get_repository_insights(
                repository="owner/repo",
                days_back=30
            )
            
            assert insights["repository"] == "owner/repo"
            assert insights["total_issues"] == len(sample_github_issues)
            assert insights["open_issues"] == 1  # sample_github_issues[1] is open
            assert insights["closed_issues"] == 2  # sample_github_issues[0] and [2] are closed
            assert "priority_distribution" in insights
            assert "incident_type_distribution" in insights
            assert "avg_resolution_time_hours" in insights
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, github_issues_searcher):
        """検索エラーハンドリングテスト"""
        with patch.object(github_issues_searcher.github_client, 'search_issues', 
                         side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                await github_issues_searcher.search_similar_issues(
                    description="Test",
                    keywords=["test"],
                    repositories=["owner/repo"]  # 単一リポジトリを指定
                )
    
    def test_to_dict(self, github_issues_searcher, sample_github_issues):
        """辞書変換テスト"""
        # サンプルレポート作成
        report = GitHubSearchReport(
            query_description="Test query",
            total_issues_found=3,
            similar_issues=[
                SimilarIssue(
                    issue=sample_github_issues[0],
                    similarity_score=0.8,
                    matching_keywords=["server"],
                    relevance_reason="High similarity"
                )
            ],
            incident_patterns=[
                IncidentPattern(
                    incident_type="outage",
                    frequency=1,
                    avg_resolution_time=8.0,
                    common_labels=["critical"],
                    frequent_assignees=["john_doe"],
                    success_patterns=["quick fix"]
                )
            ],
            suggested_solutions=["Fix server"],
            recommended_assignees=["john_doe"],
            estimated_resolution_time=12.0,
            priority_recommendation=IssuePriority.HIGH,
            timestamp=datetime.now()
        )
        
        result = github_issues_searcher.to_dict(report)
        
        assert isinstance(result, dict)
        assert result["query_description"] == "Test query"
        assert result["total_issues_found"] == 3
        assert len(result["similar_issues"]) == 1
        assert len(result["incident_patterns"]) == 1
        assert result["priority_recommendation"] == "high"
        assert "timestamp" in result


class TestSimilarIssue:
    """SimilarIssue テスト"""
    
    def test_similar_issue_creation(self, sample_github_issues):
        """SimilarIssue作成テスト"""
        similar_issue = SimilarIssue(
            issue=sample_github_issues[0],
            similarity_score=0.75,
            matching_keywords=["server", "critical"],
            relevance_reason="Keywords match"
        )
        
        assert similar_issue.similarity_score == 0.75
        assert "server" in similar_issue.matching_keywords
        assert similar_issue.relevance_reason == "Keywords match"


class TestIncidentPattern:
    """IncidentPattern テスト"""
    
    def test_incident_pattern_creation(self):
        """IncidentPattern作成テスト"""
        pattern = IncidentPattern(
            incident_type="performance",
            frequency=5,
            avg_resolution_time=24.5,
            common_labels=["performance", "api"],
            frequent_assignees=["dev1", "dev2"],
            success_patterns=["escalate to senior dev"]
        )
        
        assert pattern.incident_type == "performance"
        assert pattern.frequency == 5
        assert pattern.avg_resolution_time == 24.5
        assert len(pattern.common_labels) == 2
        assert len(pattern.frequent_assignees) == 2


@pytest.mark.asyncio
async def test_integration_with_github_client(github_issues_searcher, sample_github_issues):
    """GitHubクライアントとの統合テスト"""
    with patch.multiple(
        github_issues_searcher.github_client,
        verify_connection=AsyncMock(return_value=True),
        get_rate_limit_info=AsyncMock(return_value={"search": {"remaining": 30}}),
        search_issues=AsyncMock(return_value=sample_github_issues)
    ):
        # セットアップ確認
        setup_ok = await github_issues_searcher.verify_setup()
        assert setup_ok is True
        
        # 検索実行
        report = await github_issues_searcher.search_similar_issues(
            description="Performance issue with database",
            keywords=["performance", "database", "slow"]
        )
        
        assert isinstance(report, GitHubSearchReport)
        assert report.total_issues_found > 0


@pytest.mark.asyncio
async def test_empty_search_results(github_issues_searcher):
    """空の検索結果テスト"""
    with patch.object(github_issues_searcher.github_client, 'search_issues', return_value=[]):
        report = await github_issues_searcher.search_similar_issues(
            description="Non-existent issue",
            keywords=["nonexistent"]
        )
        
        assert report.total_issues_found == 0
        assert len(report.similar_issues) == 0
        assert len(report.incident_patterns) == 0
        assert len(report.suggested_solutions) > 0  # デフォルトの解決策が提案される


@pytest.mark.asyncio  
async def test_rate_limit_warning(github_issues_searcher):
    """レート制限警告テスト"""
    with patch.object(github_issues_searcher.github_client, 'verify_connection', return_value=True):
        with patch.object(github_issues_searcher.github_client, 'get_rate_limit_info', return_value={
            "search": {"remaining": 2}  # 低いレート制限
        }):
            setup_ok = await github_issues_searcher.verify_setup()
            assert setup_ok is True  # 警告は出るが、セットアップは成功
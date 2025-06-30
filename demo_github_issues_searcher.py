#!/usr/bin/env python3
"""
AALS Module 5: GitHub Issues Searcher デモ
インシデント類似検索・パターン分析機能の動作確認
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from aals.modules.github_issues_searcher import GitHubIssuesSearcher
from aals.integrations.github_client import GitHubIssue, IssuePriority
from aals.core.logger import get_logger


logger = get_logger(__name__)


def create_sample_issues():
    """サンプルIssueデータ作成"""
    now = datetime.now()
    
    return [
        # Critical production outage
        GitHubIssue(
            number=1001,
            title="CRITICAL: Production API completely down",
            body="All API endpoints are returning 500 errors. Users cannot access the application. This started 30 minutes ago after the latest deployment.",
            state="closed",
            labels=["critical", "production", "outage", "p0"],
            assignees=["senior-dev", "devops-lead"],
            created_at=now - timedelta(days=7),
            updated_at=now - timedelta(days=6, hours=22),
            closed_at=now - timedelta(days=6, hours=22),
            author="monitoring-bot",
            comments_count=25,
            url="https://github.com/company/backend/issues/1001",
            repository="company/backend",
            priority=IssuePriority.CRITICAL,
            incident_type="outage",
            resolution_time=timedelta(hours=2)
        ),
        
        # Database performance issue
        GitHubIssue(
            number=1002,
            title="Database queries timing out during peak hours",
            body="Multiple database queries are timing out during peak usage periods (2-4 PM). This is affecting user experience and causing some requests to fail.",
            state="closed",
            labels=["performance", "database", "high-priority"],
            assignees=["db-admin", "backend-dev"],
            created_at=now - timedelta(days=14),
            updated_at=now - timedelta(days=12),
            closed_at=now - timedelta(days=12),
            author="user-reports",
            comments_count=18,
            url="https://github.com/company/backend/issues/1002",
            repository="company/backend",
            priority=IssuePriority.HIGH,
            incident_type="performance",
            resolution_time=timedelta(hours=36)
        ),
        
        # Similar API performance issue (current)
        GitHubIssue(
            number=1003,
            title="API response times degraded",
            body="API response times have increased significantly over the past 2 hours. Some endpoints are taking 5+ seconds to respond.",
            state="open",
            labels=["performance", "api", "investigating"],
            assignees=["backend-dev"],
            created_at=now - timedelta(hours=3),
            updated_at=now - timedelta(minutes=30),
            closed_at=None,
            author="monitoring-system",
            comments_count=8,
            url="https://github.com/company/backend/issues/1003",
            repository="company/backend",
            priority=IssuePriority.HIGH,
            incident_type="performance"
        ),
        
        # Memory leak issue
        GitHubIssue(
            number=1004,
            title="Memory usage continuously increasing",
            body="Application memory usage is gradually increasing over time, eventually leading to OOM errors. This pattern started appearing after the v2.1.0 release.",
            state="closed",
            labels=["bug", "memory-leak", "p1"],
            assignees=["senior-dev"],
            created_at=now - timedelta(days=21),
            updated_at=now - timedelta(days=18),
            closed_at=now - timedelta(days=18),
            author="platform-team",
            comments_count=32,
            url="https://github.com/company/backend/issues/1004",
            repository="company/backend",
            priority=IssuePriority.HIGH,
            incident_type="bug",
            resolution_time=timedelta(hours=72)
        ),
        
        # Security vulnerability
        GitHubIssue(
            number=1005,
            title="Potential SQL injection in user search",
            body="Security audit revealed potential SQL injection vulnerability in the user search functionality. Input validation appears to be insufficient.",
            state="closed",
            labels=["security", "vulnerability", "critical"],
            assignees=["security-team", "senior-dev"],
            created_at=now - timedelta(days=45),
            updated_at=now - timedelta(days=43),
            closed_at=now - timedelta(days=43),
            author="security-audit",
            comments_count=15,
            url="https://github.com/company/backend/issues/1005",
            repository="company/backend",
            priority=IssuePriority.CRITICAL,
            incident_type="security",
            resolution_time=timedelta(hours=48)
        ),
        
        # Deployment issue
        GitHubIssue(
            number=1006,
            title="Deployment failing - container startup errors",
            body="Latest deployment is failing with container startup errors. The application won't start properly in the production environment.",
            state="open",
            labels=["deployment", "infrastructure", "p1"],
            assignees=["devops-lead"],
            created_at=now - timedelta(hours=6),
            updated_at=now - timedelta(hours=2),
            closed_at=None,
            author="ci-cd-system",
            comments_count=12,
            url="https://github.com/company/backend/issues/1006",
            repository="company/backend",
            priority=IssuePriority.HIGH,
            incident_type="deployment"
        ),
        
        # Monitoring alert
        GitHubIssue(
            number=1007,
            title="High CPU usage on web servers",
            body="Web server CPU usage has been consistently above 80% for the past hour. This is impacting response times and user experience.",
            state="closed",
            labels=["monitoring", "performance", "infrastructure"],
            assignees=["devops-lead", "backend-dev"],
            created_at=now - timedelta(days=3),
            updated_at=now - timedelta(days=2, hours=20),
            closed_at=now - timedelta(days=2, hours=20),
            author="monitoring-alert",
            comments_count=9,
            url="https://github.com/company/backend/issues/1007",
            repository="company/backend",
            priority=IssuePriority.MEDIUM,
            incident_type="monitoring",
            resolution_time=timedelta(hours=4)
        )
    ]


async def demo_github_issues_searcher():
    """GitHub Issues Searcher デモ実行"""
    print("🔍 AALS Module 5: GitHub Issues Searcher デモ")
    print("=" * 50)
    
    # 1. モジュール初期化
    print("\n📋 1. GitHub Issues Searcher 初期化")
    print("-" * 30)
    
    searcher = GitHubIssuesSearcher()
    print(f"✅ 初期化完了")
    print(f"✅ デフォルトリポジトリ: {len(searcher.default_repositories)}個")
    print(f"✅ 最大検索結果: {searcher.max_results}件")
    print(f"✅ 類似度閾値: {searcher.similarity_threshold}")
    
    # 2. セットアップ確認（モック）
    print(f"\n🔗 2. セットアップ確認")
    print("-" * 30)
    
    with patch.object(searcher.github_client, 'verify_connection', return_value=True):
        with patch.object(searcher.github_client, 'get_rate_limit_info', return_value={
            "search": {"remaining": 30, "limit": 30}
        }):
            setup_ok = await searcher.verify_setup()
            print(f"✅ セットアップ確認: {'成功' if setup_ok else '失敗'}")
            print(f"✅ GitHub API接続: OK")
            print(f"✅ レート制限残り: 30/30")
    
    # 3. サンプルデータ準備
    print(f"\n📊 3. サンプルデータ準備")
    print("-" * 30)
    
    sample_issues = create_sample_issues()
    print(f"✅ サンプルIssue生成: {len(sample_issues)}件")
    
    # データ内訳表示
    open_count = len([i for i in sample_issues if i.state == "open"])
    closed_count = len([i for i in sample_issues if i.state == "closed"])
    print(f"   📂 Open: {open_count}件, Closed: {closed_count}件")
    
    # 優先度分布
    priority_counts = {}
    for issue in sample_issues:
        priority = issue.priority.value
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    print(f"   🎯 優先度分布: {dict(sorted(priority_counts.items()))}")
    
    # 4. 類似Issue検索デモ
    print(f"\n🔍 4. 類似Issue検索デモ")
    print("-" * 30)
    
    # シナリオ1: API パフォーマンス問題
    print(f"🎯 シナリオ1: API パフォーマンス問題")
    
    with patch.object(searcher.github_client, 'search_issues', return_value=sample_issues):
        report1 = await searcher.search_similar_issues(
            description="API endpoints are responding slowly, response times have increased",
            keywords=["api", "performance", "slow", "response", "time"],
            repositories=["company/backend"]
        )
        
        print(f"   📊 検索結果: {report1.total_issues_found}件")
        print(f"   🎯 類似Issue: {len(report1.similar_issues)}件")
        print(f"   📈 パターン: {len(report1.incident_patterns)}個")
        print(f"   💡 解決策: {len(report1.suggested_solutions)}個")
        print(f"   👥 推奨担当者: {', '.join(report1.recommended_assignees) if report1.recommended_assignees else 'なし'}")
        print(f"   ⏱️  予想解決時間: {report1.estimated_resolution_time:.1f}時間" if report1.estimated_resolution_time else "   ⏱️  予想解決時間: 不明")
        print(f"   🚨 推奨優先度: {report1.priority_recommendation.value.upper()}")
        
        # 最も類似度の高いIssue
        if report1.similar_issues:
            best_match = report1.similar_issues[0]
            print(f"   🎯 最高類似Issue: #{best_match.issue.number} (類似度: {best_match.similarity_score:.2f})")
            print(f"      タイトル: {best_match.issue.title}")
            print(f"      マッチキーワード: {', '.join(best_match.matching_keywords)}")
    
    # シナリオ2: 本番環境障害
    print(f"\n🎯 シナリオ2: 本番環境障害")
    
    with patch.object(searcher.github_client, 'search_issues', return_value=sample_issues):
        report2 = await searcher.search_similar_issues(
            description="Production system is down, users cannot access the application",
            keywords=["production", "down", "outage", "critical", "users"],
            repositories=["company/backend"],
            include_closed=True
        )
        
        print(f"   📊 検索結果: {report2.total_issues_found}件")
        print(f"   🎯 類似Issue: {len(report2.similar_issues)}件")
        print(f"   🚨 推奨優先度: {report2.priority_recommendation.value.upper()}")
        
        # 解決済みの類似Issue
        resolved_similar = [si for si in report2.similar_issues if si.issue.state == "closed"]
        print(f"   ✅ 解決済み類似Issue: {len(resolved_similar)}件")
        
        if resolved_similar:
            fastest_resolution = min(resolved_similar, key=lambda x: x.issue.resolution_time or timedelta(days=999))
            print(f"   ⚡ 最速解決事例: #{fastest_resolution.issue.number}")
            print(f"      解決時間: {fastest_resolution.issue.resolution_time}")
            print(f"      担当者: {', '.join(fastest_resolution.issue.assignees)}")
    
    # 5. インシデントパターン分析
    print(f"\n📈 5. インシデントパターン分析")
    print("-" * 30)
    
    if report1.incident_patterns:
        print(f"✅ 検出パターン数: {len(report1.incident_patterns)}個")
        
        for i, pattern in enumerate(report1.incident_patterns, 1):
            print(f"   📊 パターン {i}: {pattern.incident_type}")
            print(f"      頻度: {pattern.frequency}件")
            print(f"      平均解決時間: {pattern.avg_resolution_time:.1f}時間")
            print(f"      よく使われるラベル: {', '.join(pattern.common_labels[:3])}")
            print(f"      頻出担当者: {', '.join(pattern.frequent_assignees[:2])}")
            if pattern.success_patterns:
                print(f"      成功パターン: {', '.join(pattern.success_patterns[:2])}")
    
    # 6. 解決策提案
    print(f"\n💡 6. 解決策提案")
    print("-" * 30)
    
    if report1.suggested_solutions:
        for i, solution in enumerate(report1.suggested_solutions, 1):
            print(f"💡 提案 {i}: {solution}")
    
    # 7. リポジトリインサイト
    print(f"\n📊 7. リポジトリインサイト")
    print("-" * 30)
    
    with patch.object(searcher.github_client, 'get_repository_issues', return_value=sample_issues):
        insights = await searcher.get_repository_insights(
            repository="company/backend",
            days_back=30
        )
        
        print(f"✅ リポジトリ: {insights['repository']}")
        print(f"📊 総Issue数: {insights['total_issues']}件")
        print(f"📂 Open: {insights['open_issues']}件, Closed: {insights['closed_issues']}件")
        print(f"✅ 解決率: {insights['close_rate']:.1f}%")
        print(f"⏱️  平均解決時間: {insights['avg_resolution_time_hours']:.1f}時間")
        
        # 優先度分布
        print(f"🎯 優先度分布:")
        for priority, count in insights['priority_distribution'].items():
            print(f"   {priority}: {count}件")
        
        # インシデントタイプ分布
        print(f"🔍 インシデントタイプ分布:")
        for incident_type, count in insights['incident_type_distribution'].items():
            print(f"   {incident_type}: {count}件")
    
    # 8. 類似度計算デモ
    print(f"\n🔬 8. 類似度計算デモ")
    print("-" * 30)
    
    test_cases = [
        ("API response slow", "API endpoints responding slowly"),
        ("Database timeout", "Database queries timing out"),
        ("Production down", "Server maintenance scheduled"),
        ("Security issue", "Performance optimization needed")
    ]
    
    for text1, text2 in test_cases:
        similarity = searcher._calculate_text_similarity(text1, text2)
        print(f"📊 '{text1}' vs '{text2}': {similarity:.2f}")
    
    # 9. エラーハンドリングデモ
    print(f"\n🚨 9. エラーハンドリングデモ")
    print("-" * 30)
    
    with patch.object(searcher.github_client, 'search_issues', side_effect=Exception("API Rate Limited")):
        try:
            await searcher.search_similar_issues(
                description="Test error handling",
                keywords=["test"]
            )
        except Exception as e:
            print(f"✅ エラーハンドリング確認: {str(e)}")
    
    # 10. パフォーマンス統計
    print(f"\n⚡ 10. パフォーマンス統計")
    print("-" * 30)
    
    print(f"✅ 検索処理: {len(sample_issues)}件のIssue分析")
    print(f"✅ 類似度計算: {len(sample_issues)}回実行")
    print(f"✅ パターン分析: {len(report1.incident_patterns)}個のパターン検出")
    print(f"✅ 解決策生成: {len(report1.suggested_solutions)}個の提案")
    
    # 監査ログ確認
    from aals.core.logger import AuditAction, AuditLogEntry, audit_log
    
    audit_log(AuditLogEntry(
        action=AuditAction.VIEW,
        resource="github_issues_demo",
        result="success",
        details=f"GitHub Issues Searcher デモ完了 - {len(sample_issues)}件分析",
        risk_level="low"
    ))
    
    logger.info("GitHub Issues Searcher demo completed",
                total_issues=len(sample_issues),
                scenarios_tested=2,
                patterns_found=len(report1.incident_patterns))
    
    print(f"\n🎉 Module 5: GitHub Issues Searcher デモ完了!")
    print(f"   🔍 検索シナリオ: 2件実行")
    print(f"   📊 分析Issue数: {len(sample_issues)}件")
    print(f"   📈 パターン検出: {len(report1.incident_patterns)}個")
    print(f"   💡 解決策提案: {len(report1.suggested_solutions)}個")
    print(f"   📊 リポジトリインサイト: 1件生成")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(demo_github_issues_searcher())
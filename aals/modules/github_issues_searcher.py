#!/usr/bin/env python3
"""
AALS Module 5: GitHub Issues Searcher
GitHubからインシデント・Issue情報を検索・分析し、類似事例や解決策を発見
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from aals.core.config import get_config_manager
from aals.core.logger import get_logger, AuditAction, AuditLogEntry, audit_log
from aals.integrations.github_client import (
    GitHubAPIClient, 
    GitHubIssue, 
    SearchQuery, 
    IssueState, 
    IssuePriority
)


logger = get_logger(__name__)


@dataclass
class SimilarIssue:
    """類似Issue情報"""
    issue: GitHubIssue
    similarity_score: float
    matching_keywords: List[str]
    relevance_reason: str


@dataclass
class IncidentPattern:
    """インシデントパターン分析結果"""
    incident_type: str
    frequency: int
    avg_resolution_time: float
    common_labels: List[str]
    frequent_assignees: List[str]
    success_patterns: List[str]


@dataclass
class GitHubSearchReport:
    """GitHub検索レポート"""
    query_description: str
    total_issues_found: int
    similar_issues: List[SimilarIssue]
    incident_patterns: List[IncidentPattern]
    suggested_solutions: List[str]
    recommended_assignees: List[str]
    estimated_resolution_time: Optional[float]
    priority_recommendation: IssuePriority
    timestamp: datetime


class GitHubIssuesSearcher:
    """GitHub Issues検索・分析モジュール"""
    
    def __init__(self):
        """初期化"""
        config_manager = get_config_manager()
        module_config = config_manager.get_module_config("github_issues_searcher")
        
        if not module_config.enabled:
            raise RuntimeError("GitHub Issues Searcher module is not enabled")
        
        self.config = module_config.config
        self.github_token = self.config.get("token")
        self.default_repositories = self.config.get("repositories", [])
        self.search_timeout = self.config.get("timeout", 30)
        self.max_results = self.config.get("max_results", 50)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.3)
        
        # GitHubクライアント初期化
        self.github_client = GitHubAPIClient(
            token=self.github_token,
            timeout=self.search_timeout
        )
        
        logger.info("GitHub Issues Searcher initialized",
                   has_token=bool(self.github_token),
                   repositories=len(self.default_repositories),
                   max_results=self.max_results)
    
    async def verify_setup(self) -> bool:
        """セットアップ確認"""
        try:
            # GitHub API接続確認
            connection_ok = await self.github_client.verify_connection()
            
            if not connection_ok:
                logger.error("GitHub API connection failed")
                return False
            
            # レート制限確認
            rate_info = await self.github_client.get_rate_limit_info()
            remaining_searches = rate_info["search"]["remaining"]
            
            if remaining_searches < 5:
                logger.warning("GitHub search rate limit low", 
                             remaining=remaining_searches)
            
            logger.info("GitHub Issues Searcher setup verified",
                       rate_limit_remaining=remaining_searches)
            
            return True
            
        except Exception as e:
            logger.error("GitHub Issues Searcher setup verification failed",
                        error=str(e))
            return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """テキスト類似度計算（簡易版）"""
        if not text1 or not text2:
            return 0.0
        
        # 単語レベルでの類似度計算
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _find_matching_keywords(self, target_text: str, issue: GitHubIssue) -> List[str]:
        """マッチするキーワードを検索"""
        target_words = set(target_text.lower().split())
        issue_text = f"{issue.title} {issue.body}".lower()
        issue_words = set(issue_text.split())
        
        # 共通キーワード
        common_words = target_words.intersection(issue_words)
        
        # フィルタリング（一般的すぎる単語を除外）
        filtered_keywords = [
            word for word in common_words 
            if len(word) > 3 and word not in {"the", "and", "for", "with", "that", "this"}
        ]
        
        return sorted(filtered_keywords)
    
    def _analyze_incident_patterns(self, issues: List[GitHubIssue]) -> List[IncidentPattern]:
        """インシデントパターン分析"""
        if not issues:
            return []
        
        # インシデントタイプ別グループ化
        patterns_data = {}
        
        for issue in issues:
            incident_type = issue.incident_type or "unknown"
            
            if incident_type not in patterns_data:
                patterns_data[incident_type] = {
                    "issues": [],
                    "resolution_times": [],
                    "labels": [],
                    "assignees": []
                }
            
            patterns_data[incident_type]["issues"].append(issue)
            patterns_data[incident_type]["labels"].extend(issue.labels)
            patterns_data[incident_type]["assignees"].extend(issue.assignees)
            
            if issue.resolution_time:
                resolution_hours = issue.resolution_time.total_seconds() / 3600
                patterns_data[incident_type]["resolution_times"].append(resolution_hours)
        
        # パターン生成
        patterns = []
        for incident_type, data in patterns_data.items():
            if len(data["issues"]) < 2:  # 最低2件以上
                continue
            
            # 平均解決時間
            avg_resolution = (
                sum(data["resolution_times"]) / len(data["resolution_times"])
                if data["resolution_times"] else 0.0
            )
            
            # よく使われるラベル
            label_counts = {}
            for label in data["labels"]:
                label_counts[label] = label_counts.get(label, 0) + 1
            common_labels = sorted(label_counts.keys(), key=label_counts.get, reverse=True)[:5]
            
            # よく担当するassignee
            assignee_counts = {}
            for assignee in data["assignees"]:
                assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
            frequent_assignees = sorted(assignee_counts.keys(), key=assignee_counts.get, reverse=True)[:3]
            
            # 成功パターン（解決済みのIssueから）
            success_patterns = []
            closed_issues = [issue for issue in data["issues"] if issue.state == "closed"]
            if closed_issues:
                # 解決時間が短いIssueの特徴を抽出
                fast_resolved = sorted(closed_issues, key=lambda x: x.resolution_time or timedelta(days=999))[:3]
                for issue in fast_resolved:
                    if issue.labels:
                        success_patterns.append(f"Label: {', '.join(issue.labels[:2])}")
                    if issue.assignees:
                        success_patterns.append(f"Assignee: {issue.assignees[0]}")
            
            pattern = IncidentPattern(
                incident_type=incident_type,
                frequency=len(data["issues"]),
                avg_resolution_time=avg_resolution,
                common_labels=common_labels,
                frequent_assignees=frequent_assignees,
                success_patterns=list(set(success_patterns))
            )
            patterns.append(pattern)
        
        return sorted(patterns, key=lambda x: x.frequency, reverse=True)
    
    def _generate_solutions(self, similar_issues: List[SimilarIssue], patterns: List[IncidentPattern]) -> List[str]:
        """解決策提案生成"""
        solutions = []
        
        # 類似Issueから解決策抽出
        closed_similar = [si for si in similar_issues if si.issue.state == "closed"]
        if closed_similar:
            solutions.append("Similar resolved issues found - review their resolution approaches")
            
            # 高類似度で解決済みのIssueから具体的提案
            high_similarity = [si for si in closed_similar if si.similarity_score > 0.5]
            if high_similarity:
                best_match = max(high_similarity, key=lambda x: x.similarity_score)
                solutions.append(f"High confidence match: {best_match.issue.title} (#{best_match.issue.number})")
        
        # パターンから解決策
        if patterns:
            main_pattern = patterns[0]  # 最頻出パターン
            if main_pattern.success_patterns:
                solutions.append(f"Apply successful {main_pattern.incident_type} patterns: {', '.join(main_pattern.success_patterns[:2])}")
            
            if main_pattern.frequent_assignees:
                solutions.append(f"Consider assigning to experienced team member: {main_pattern.frequent_assignees[0]}")
        
        # 一般的な推奨事項
        if not solutions:
            solutions.extend([
                "Check system logs and monitoring dashboards",
                "Review recent deployments and configuration changes",
                "Escalate to on-call engineer if critical"
            ])
        
        return solutions[:5]  # 最大5個
    
    def _recommend_assignees(self, patterns: List[IncidentPattern], issue_labels: List[str]) -> List[str]:
        """担当者推奨"""
        recommended = []
        
        # パターンマッチングによる推奨
        for pattern in patterns:
            # ラベルマッチする担当者
            if any(label in pattern.common_labels for label in issue_labels):
                recommended.extend(pattern.frequent_assignees)
        
        # 重複除去・優先度順
        unique_recommended = []
        seen = set()
        for assignee in recommended:
            if assignee not in seen:
                unique_recommended.append(assignee)
                seen.add(assignee)
        
        return unique_recommended[:3]  # 最大3人
    
    def _estimate_resolution_time(self, patterns: List[IncidentPattern], issue_priority: IssuePriority) -> Optional[float]:
        """解決時間予測"""
        if not patterns:
            return None
        
        # 優先度別の基本時間
        base_times = {
            IssuePriority.CRITICAL: 2.0,    # 2時間
            IssuePriority.HIGH: 8.0,        # 8時間
            IssuePriority.MEDIUM: 24.0,     # 1日
            IssuePriority.LOW: 72.0,        # 3日
            IssuePriority.UNKNOWN: 48.0     # 2日
        }
        
        base_time = base_times.get(issue_priority, 48.0)
        
        # パターンデータで調整
        if patterns:
            pattern_avg = sum(p.avg_resolution_time for p in patterns) / len(patterns)
            if pattern_avg > 0:
                # 基本時間とパターン平均の重み付き平均
                estimated = (base_time * 0.3) + (pattern_avg * 0.7)
                return estimated
        
        return base_time
    
    def _recommend_priority(self, similar_issues: List[SimilarIssue], keywords: List[str]) -> IssuePriority:
        """優先度推奨"""
        # キーワードベースの判定
        critical_keywords = ["down", "outage", "critical", "emergency", "production"]
        high_keywords = ["urgent", "important", "blocker", "security"]
        
        text_to_check = " ".join(keywords).lower()
        
        if any(keyword in text_to_check for keyword in critical_keywords):
            return IssuePriority.CRITICAL
        elif any(keyword in text_to_check for keyword in high_keywords):
            return IssuePriority.HIGH
        
        # 類似Issueの優先度を参考
        if similar_issues:
            priorities = [si.issue.priority for si in similar_issues if si.issue.priority != IssuePriority.UNKNOWN]
            if priorities:
                critical_count = sum(1 for p in priorities if p == IssuePriority.CRITICAL)
                high_count = sum(1 for p in priorities if p == IssuePriority.HIGH)
                
                if critical_count > len(priorities) * 0.3:
                    return IssuePriority.HIGH  # Critical多い→High推奨
                elif high_count > len(priorities) * 0.5:
                    return IssuePriority.MEDIUM
        
        return IssuePriority.MEDIUM  # デフォルト
    
    async def search_similar_issues(
        self,
        description: str,
        keywords: List[str],
        repositories: Optional[List[str]] = None,
        hours_back: Optional[int] = None,
        include_closed: bool = True
    ) -> GitHubSearchReport:
        """類似Issue検索・分析"""
        try:
            logger.info("Starting GitHub issues search",
                       keywords=keywords,
                       repositories=repositories or self.default_repositories,
                       hours_back=hours_back)
            
            # 検索クエリ構築
            search_repos = repositories or self.default_repositories
            all_issues = []
            
            # リポジトリ別検索
            for repo in search_repos:
                try:
                    # キーワードベースの検索
                    query = SearchQuery(
                        keywords=keywords,
                        repository=repo,
                        state=IssueState.ALL if include_closed else IssueState.OPEN,
                        created_after=datetime.now() - timedelta(hours=hours_back) if hours_back else None
                    )
                    
                    issues = await self.github_client.search_issues(query, max_results=self.max_results)
                    all_issues.extend(issues)
                    
                    logger.debug("Repository search completed",
                               repository=repo, issues_found=len(issues))
                    
                except Exception as e:
                    logger.warning("Repository search failed",
                                 repository=repo, error=str(e))
                    continue
            
            logger.info("GitHub search completed", total_issues=len(all_issues))
            
            # 類似度計算
            similar_issues = []
            target_text = f"{description} {' '.join(keywords)}"
            
            for issue in all_issues:
                issue_text = f"{issue.title} {issue.body}"
                similarity = self._calculate_text_similarity(target_text, issue_text)
                
                if similarity >= self.similarity_threshold:
                    matching_keywords = self._find_matching_keywords(target_text, issue)
                    
                    # 関連性の理由
                    relevance_reason = f"Text similarity: {similarity:.2f}"
                    if matching_keywords:
                        relevance_reason += f", Keywords: {', '.join(matching_keywords[:3])}"
                    
                    similar_issue = SimilarIssue(
                        issue=issue,
                        similarity_score=similarity,
                        matching_keywords=matching_keywords,
                        relevance_reason=relevance_reason
                    )
                    similar_issues.append(similar_issue)
            
            # 類似度順でソート
            similar_issues.sort(key=lambda x: x.similarity_score, reverse=True)
            similar_issues = similar_issues[:20]  # 上位20件
            
            # パターン分析
            patterns = self._analyze_incident_patterns([si.issue for si in similar_issues])
            
            # 解決策生成
            solutions = self._generate_solutions(similar_issues, patterns)
            
            # 担当者推奨
            recommended_assignees = self._recommend_assignees(patterns, keywords)
            
            # 解決時間予測
            priority_rec = self._recommend_priority(similar_issues, keywords)
            estimated_time = self._estimate_resolution_time(patterns, priority_rec)
            
            # レポート生成
            report = GitHubSearchReport(
                query_description=description,
                total_issues_found=len(all_issues),
                similar_issues=similar_issues,
                incident_patterns=patterns,
                suggested_solutions=solutions,
                recommended_assignees=recommended_assignees,
                estimated_resolution_time=estimated_time,
                priority_recommendation=priority_rec,
                timestamp=datetime.now()
            )
            
            # 監査ログ
            audit_log(AuditLogEntry(
                action=AuditAction.VIEW,
                resource="github_issues",
                result="success",
                details=f"{len(similar_issues)}件の類似Issue発見",
                risk_level="low"
            ))
            
            logger.info("GitHub issues search analysis completed",
                       similar_issues=len(similar_issues),
                       patterns=len(patterns),
                       solutions=len(solutions))
            
            return report
            
        except Exception as e:
            logger.error("GitHub issues search failed", error=str(e))
            audit_log(AuditLogEntry(
                action=AuditAction.VIEW,
                resource="github_issues",
                result="failure",
                details=f"検索エラー: {str(e)}",
                risk_level="medium"
            ))
            raise
    
    async def get_repository_insights(self, repository: str, days_back: int = 30) -> Dict[str, Any]:
        """リポジトリインサイト取得"""
        try:
            since = datetime.now() - timedelta(days=days_back)
            
            # Issue取得
            issues = await self.github_client.get_repository_issues(
                repository=repository,
                state=IssueState.ALL,
                since=since,
                max_results=200
            )
            
            if not issues:
                return {"repository": repository, "insights": "No issues found"}
            
            # 統計計算
            total_issues = len(issues)
            open_issues = len([i for i in issues if i.state == "open"])
            closed_issues = total_issues - open_issues
            
            # 優先度分布
            priority_dist = {}
            for issue in issues:
                priority = issue.priority.value
                priority_dist[priority] = priority_dist.get(priority, 0) + 1
            
            # インシデントタイプ分布
            incident_dist = {}
            for issue in issues:
                incident_type = issue.incident_type or "unknown"
                incident_dist[incident_type] = incident_dist.get(incident_type, 0) + 1
            
            # 平均解決時間
            resolution_times = [
                issue.resolution_time.total_seconds() / 3600 
                for issue in issues 
                if issue.resolution_time
            ]
            avg_resolution = sum(resolution_times) / len(resolution_times) if resolution_times else 0
            
            insights = {
                "repository": repository,
                "period_days": days_back,
                "total_issues": total_issues,
                "open_issues": open_issues,
                "closed_issues": closed_issues,
                "close_rate": (closed_issues / total_issues * 100) if total_issues > 0 else 0,
                "priority_distribution": priority_dist,
                "incident_type_distribution": incident_dist,
                "avg_resolution_time_hours": round(avg_resolution, 2),
                "most_active_contributors": [
                    issue.author for issue in issues[:10]  # 上位10人
                ],
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info("Repository insights generated",
                       repository=repository, total_issues=total_issues)
            
            return insights
            
        except Exception as e:
            logger.error("Repository insights generation failed",
                        repository=repository, error=str(e))
            raise
    
    def to_dict(self, report: GitHubSearchReport) -> Dict[str, Any]:
        """レポートを辞書に変換"""
        return {
            "query_description": report.query_description,
            "total_issues_found": report.total_issues_found,
            "similar_issues": [
                {
                    "issue": si.issue.to_dict(),
                    "similarity_score": si.similarity_score,
                    "matching_keywords": si.matching_keywords,
                    "relevance_reason": si.relevance_reason
                }
                for si in report.similar_issues
            ],
            "incident_patterns": [
                {
                    "incident_type": pattern.incident_type,
                    "frequency": pattern.frequency,
                    "avg_resolution_time_hours": pattern.avg_resolution_time,
                    "common_labels": pattern.common_labels,
                    "frequent_assignees": pattern.frequent_assignees,
                    "success_patterns": pattern.success_patterns
                }
                for pattern in report.incident_patterns
            ],
            "suggested_solutions": report.suggested_solutions,
            "recommended_assignees": report.recommended_assignees,
            "estimated_resolution_time_hours": report.estimated_resolution_time,
            "priority_recommendation": report.priority_recommendation.value,
            "timestamp": report.timestamp.isoformat()
        }
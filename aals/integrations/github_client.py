#!/usr/bin/env python3
"""
AALS GitHub API クライアント
GitHubからIssues・PR・インシデント情報を検索・分析
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from github import Github, Issue, PullRequest, Repository
from github.GithubException import GithubException
from pydantic import BaseModel, Field, ConfigDict

from aals.core.logger import get_logger


logger = get_logger(__name__)


class IssueState(Enum):
    """Issues状態"""
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


class IssuePriority(Enum):
    """Issue優先度"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class GitHubIssue:
    """GitHubのIssue情報"""
    number: int
    title: str
    body: str
    state: str
    labels: List[str]
    assignees: List[str]
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime]
    author: str
    comments_count: int
    url: str
    repository: str
    
    # 分析用フィールド
    priority: IssuePriority = IssuePriority.UNKNOWN
    incident_type: Optional[str] = None
    resolution_time: Optional[timedelta] = None
    
    def __post_init__(self):
        """初期化後処理"""
        self.priority = self._extract_priority()
        self.incident_type = self._extract_incident_type()
        
        if self.closed_at and self.created_at:
            self.resolution_time = self.closed_at - self.created_at
    
    def _extract_priority(self) -> IssuePriority:
        """ラベルから優先度を抽出"""
        priority_labels = {
            "critical": IssuePriority.CRITICAL,
            "high": IssuePriority.HIGH,
            "medium": IssuePriority.MEDIUM,
            "low": IssuePriority.LOW,
            "priority:critical": IssuePriority.CRITICAL,
            "priority:high": IssuePriority.HIGH,
            "priority:medium": IssuePriority.MEDIUM,
            "priority:low": IssuePriority.LOW,
            "p0": IssuePriority.CRITICAL,
            "p1": IssuePriority.HIGH,
            "p2": IssuePriority.MEDIUM,
            "p3": IssuePriority.LOW
        }
        
        for label in self.labels:
            label_lower = label.lower()
            if label_lower in priority_labels:
                return priority_labels[label_lower]
        
        # タイトル・本文からも推測
        text = f"{self.title} {self.body}".lower()
        if any(word in text for word in ["critical", "urgent", "emergency", "production down"]):
            return IssuePriority.CRITICAL
        elif any(word in text for word in ["high", "important", "blocker"]):
            return IssuePriority.HIGH
        
        return IssuePriority.UNKNOWN
    
    def _extract_incident_type(self) -> Optional[str]:
        """インシデントタイプを推測"""
        labels_text = " ".join(self.labels).lower()
        title_body = f"{self.title} {self.body}".lower()
        
        incident_keywords = {
            "outage": ["outage", "down", "unavailable", "service down"],
            "performance": ["slow", "performance", "latency", "timeout"],
            "security": ["security", "vulnerability", "breach", "exploit"],
            "bug": ["bug", "error", "exception", "crash", "fail"],
            "deployment": ["deployment", "deploy", "release", "rollback"],
            "monitoring": ["alert", "monitoring", "metrics", "dashboard"],
            "infrastructure": ["infrastructure", "server", "database", "network"]
        }
        
        for incident_type, keywords in incident_keywords.items():
            if any(keyword in labels_text or keyword in title_body for keyword in keywords):
                return incident_type
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body[:500] + "..." if len(self.body) > 500 else self.body,
            "state": self.state,
            "labels": self.labels,
            "assignees": self.assignees,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "author": self.author,
            "comments_count": self.comments_count,
            "url": self.url,
            "repository": self.repository,
            "priority": self.priority.value,
            "incident_type": self.incident_type,
            "resolution_time_hours": self.resolution_time.total_seconds() / 3600 if self.resolution_time else None
        }


@dataclass
class SearchQuery:
    """GitHub検索クエリ"""
    keywords: List[str]
    labels: List[str] = None
    state: IssueState = IssueState.ALL
    author: Optional[str] = None
    assignee: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    repository: Optional[str] = None
    
    def to_github_query(self) -> str:
        """GitHub Search API用クエリ文字列に変換"""
        query_parts = []
        
        # キーワード
        if self.keywords:
            query_parts.append(" ".join(self.keywords))
        
        # ラベル
        if self.labels:
            for label in self.labels:
                query_parts.append(f'label:"{label}"')
        
        # 状態
        if self.state != IssueState.ALL:
            query_parts.append(f"state:{self.state.value}")
        
        # 作成者
        if self.author:
            query_parts.append(f"author:{self.author}")
        
        # アサイン先
        if self.assignee:
            query_parts.append(f"assignee:{self.assignee}")
        
        # 作成日
        if self.created_after:
            query_parts.append(f"created:>={self.created_after.strftime('%Y-%m-%d')}")
        
        if self.created_before:
            query_parts.append(f"created:<={self.created_before.strftime('%Y-%m-%d')}")
        
        # リポジトリ
        if self.repository:
            query_parts.append(f"repo:{self.repository}")
        
        # Issueのみに限定
        query_parts.append("type:issue")
        
        return " ".join(query_parts)


class GitHubAPIClient:
    """GitHub API クライアント"""
    
    def __init__(self, token: Optional[str] = None, timeout: int = 30):
        """
        Args:
            token: GitHub Personal Access Token
            timeout: HTTPタイムアウト（秒）
        """
        self.token = token
        self.timeout = timeout
        self._github: Optional[Github] = None
        
        logger.info("GitHub API Client initialized", 
                   has_token=bool(token), timeout=timeout)
    
    def _get_github_client(self) -> Github:
        """GitHub クライアントを取得"""
        if not self._github:
            if self.token:
                self._github = Github(self.token, timeout=self.timeout)
            else:
                # トークンなし（レート制限あり）
                self._github = Github(timeout=self.timeout)
                logger.warning("GitHub client initialized without token - rate limited")
        
        return self._github
    
    async def verify_connection(self) -> bool:
        """GitHub API接続確認"""
        try:
            github = self._get_github_client()
            
            # Rate limit情報取得で接続確認
            rate_limit = github.get_rate_limit()
            remaining = rate_limit.core.remaining
            
            logger.info("GitHub connection verified successfully", 
                       rate_limit_remaining=remaining)
            return True
            
        except Exception as e:
            logger.error("GitHub connection verification failed", 
                        error=str(e), exception_type=type(e).__name__)
            return False
    
    def _issue_to_github_issue(self, issue: Issue) -> GitHubIssue:
        """GitHub Issue を GitHubIssue に変換"""
        return GitHubIssue(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            state=issue.state,
            labels=[label.name for label in issue.labels],
            assignees=[assignee.login for assignee in issue.assignees],
            created_at=issue.created_at,
            updated_at=issue.updated_at,
            closed_at=issue.closed_at,
            author=issue.user.login if issue.user else "unknown",
            comments_count=issue.comments,
            url=issue.html_url,
            repository=issue.repository.full_name if issue.repository else "unknown"
        )
    
    async def search_issues(
        self, 
        query: SearchQuery, 
        max_results: int = 100
    ) -> List[GitHubIssue]:
        """Issue検索"""
        try:
            github = self._get_github_client()
            
            # クエリ文字列生成
            query_string = query.to_github_query()
            logger.info("Searching GitHub issues", 
                       query=query_string, max_results=max_results)
            
            # 検索実行
            search_result = github.search_issues(query_string)
            
            issues = []
            count = 0
            
            for issue in search_result:
                if count >= max_results:
                    break
                
                github_issue = self._issue_to_github_issue(issue)
                issues.append(github_issue)
                count += 1
            
            logger.info("GitHub issues search completed", 
                       issues_found=len(issues), query=query_string)
            
            return issues
            
        except GithubException as e:
            logger.error("GitHub API error during search", 
                        error=str(e), status=e.status)
            raise
        except Exception as e:
            logger.error("GitHub search failed", 
                        error=str(e), exception_type=type(e).__name__)
            raise
    
    async def get_repository_issues(
        self, 
        repository: str, 
        state: IssueState = IssueState.ALL,
        labels: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        max_results: int = 100
    ) -> List[GitHubIssue]:
        """特定リポジトリのIssue取得"""
        try:
            github = self._get_github_client()
            repo = github.get_repo(repository)
            
            # パラメータ設定
            kwargs = {"state": state.value}
            if labels:
                kwargs["labels"] = labels
            if since:
                kwargs["since"] = since
            
            logger.info("Fetching repository issues", 
                       repository=repository, state=state.value, 
                       labels=labels, max_results=max_results)
            
            # Issue取得
            issues = []
            count = 0
            
            for issue in repo.get_issues(**kwargs):
                if count >= max_results:
                    break
                
                # PRは除外
                if issue.pull_request:
                    continue
                
                github_issue = self._issue_to_github_issue(issue)
                issues.append(github_issue)
                count += 1
            
            logger.info("Repository issues fetched", 
                       repository=repository, issues_count=len(issues))
            
            return issues
            
        except GithubException as e:
            logger.error("GitHub API error fetching repository issues", 
                        repository=repository, error=str(e))
            raise
        except Exception as e:
            logger.error("Repository issues fetch failed", 
                        repository=repository, error=str(e))
            raise
    
    async def get_issue_details(self, repository: str, issue_number: int) -> Optional[GitHubIssue]:
        """特定Issue詳細取得"""
        try:
            github = self._get_github_client()
            repo = github.get_repo(repository)
            issue = repo.get_issue(issue_number)
            
            github_issue = self._issue_to_github_issue(issue)
            
            logger.info("Issue details fetched", 
                       repository=repository, issue_number=issue_number)
            
            return github_issue
            
        except GithubException as e:
            if e.status == 404:
                logger.warning("Issue not found", 
                              repository=repository, issue_number=issue_number)
                return None
            else:
                logger.error("GitHub API error fetching issue details", 
                            repository=repository, issue_number=issue_number,
                            error=str(e))
                raise
        except Exception as e:
            logger.error("Issue details fetch failed", 
                        repository=repository, issue_number=issue_number,
                        error=str(e))
            raise
    
    async def get_rate_limit_info(self) -> Dict[str, Any]:
        """レート制限情報取得"""
        try:
            github = self._get_github_client()
            rate_limit = github.get_rate_limit()
            
            info = {
                "core": {
                    "limit": rate_limit.core.limit,
                    "remaining": rate_limit.core.remaining,
                    "reset": rate_limit.core.reset,
                    "used": rate_limit.core.used
                },
                "search": {
                    "limit": rate_limit.search.limit,
                    "remaining": rate_limit.search.remaining,
                    "reset": rate_limit.search.reset,
                    "used": rate_limit.search.used
                }
            }
            
            logger.info("Rate limit info retrieved", 
                       core_remaining=info["core"]["remaining"],
                       search_remaining=info["search"]["remaining"])
            
            return info
            
        except Exception as e:
            logger.error("Failed to get rate limit info", error=str(e))
            raise
#!/usr/bin/env python3
"""
AALS Module 8: SSH Executor
セキュアなリモートコマンド実行・自動化モジュール

PURPOSE: リモートサーバーへのSSH接続を管理し、段階的な権限制御と承認ワークフローを通じて
         安全にコマンドを実行する。全ての実行は監査ログに記録される。
DEPENDENCIES: paramiko, asyncssh
INPUT: CommandRequest (コマンド、ターゲットホスト、権限レベル)
OUTPUT: CommandResult (実行結果、ログ、メタデータ)
INTEGRATION: Module 7 (Alert Correlator) → Module 8 → Module 9 (Response Orchestrator)
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import uuid

import paramiko
import asyncssh
from asyncssh import SSHClientConnection

from aals.core.config import get_config_manager
from aals.core.logger import get_logger, AuditAction, AuditLogEntry, audit_log


logger = get_logger(__name__)


class PermissionLevel(Enum):
    """実行権限レベル"""
    READ_ONLY = "read_only"          # 読み取り専用コマンド
    LOW_RISK = "low_risk"            # 低リスクコマンド（開発環境）
    MEDIUM_RISK = "medium_risk"      # 中リスクコマンド（ステージング）
    HIGH_RISK = "high_risk"          # 高リスクコマンド（本番読み取り）
    CRITICAL = "critical"            # クリティカル（本番変更）


class ApprovalStatus(Enum):
    """承認ステータス"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


class ExecutionStatus(Enum):
    """実行ステータス"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SSHTarget:
    """SSH接続先情報"""
    host: str
    port: int = 22
    username: str = "ubuntu"
    key_file: Optional[str] = None
    password: Optional[str] = None
    environment: str = "development"  # development, staging, production
    region: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "environment": self.environment,
            "region": self.region,
            "tags": self.tags
        }


@dataclass
class CommandRequest:
    """コマンド実行リクエスト"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    command: str = ""
    targets: List[SSHTarget] = field(default_factory=list)
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    timeout_seconds: int = 300
    parallel_execution: bool = True
    dry_run: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    requested_by: str = "system"
    requested_at: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # Alert Correlatorとの連携用
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "command": self.command,
            "targets": [t.to_dict() for t in self.targets],
            "permission_level": self.permission_level.value,
            "timeout_seconds": self.timeout_seconds,
            "parallel_execution": self.parallel_execution,
            "dry_run": self.dry_run,
            "context": self.context,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "correlation_id": self.correlation_id
        }


@dataclass
class ApprovalRequest:
    """承認リクエスト"""
    approval_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    command_request: CommandRequest = None
    required_approvers: List[str] = field(default_factory=list)
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approvals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    expiry_time: datetime = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_approved(self) -> bool:
        """承認済みかチェック"""
        if self.approval_status == ApprovalStatus.AUTO_APPROVED:
            return True
        
        if self.approval_status == ApprovalStatus.APPROVED:
            return len(self.approvals) >= len(self.required_approvers)
        
        return False
    
    def is_expired(self) -> bool:
        """期限切れかチェック"""
        if self.expiry_time:
            return datetime.now() > self.expiry_time
        return False


@dataclass
class CommandResult:
    """コマンド実行結果"""
    request_id: str
    target: SSHTarget
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    status: ExecutionStatus
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "target": self.target.to_dict(),
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:1000] if len(self.stdout) > 1000 else self.stdout,  # 制限
            "stderr": self.stderr[:1000] if len(self.stderr) > 1000 else self.stderr,
            "execution_time": self.execution_time,
            "status": self.status.value,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class ExecutionReport:
    """実行レポート"""
    request_id: str
    command_request: CommandRequest
    approval_request: Optional[ApprovalRequest]
    results: List[CommandResult]
    overall_status: ExecutionStatus
    total_execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "command_request": self.command_request.to_dict(),
            "approval_status": self.approval_request.approval_status.value if self.approval_request else None,
            "results": [r.to_dict() for r in self.results],
            "overall_status": self.overall_status.value,
            "total_execution_time": self.total_execution_time,
            "success_rate": sum(1 for r in self.results if r.exit_code == 0) / len(self.results) if self.results else 0,
            "timestamp": self.timestamp.isoformat()
        }


class CommandValidator:
    """コマンドバリデーター"""
    
    # 安全なコマンドパターン
    SAFE_COMMANDS = {
        PermissionLevel.READ_ONLY: [
            r"^ls(\s|$)",
            r"^pwd$",
            r"^hostname$",
            r"^date$",
            r"^df(\s|$)",
            r"^free(\s|$)",
            r"^top\s+-b\s+-n\s+1",
            r"^ps(\s|$)",
            r"^netstat(\s|$)",
            r"^cat\s+/proc/",
            r"^tail(\s|$)",
            r"^head(\s|$)",
            r"^grep(\s|$)",
            r"^find\s+.*\s+-name",
            r"^du(\s|$)",
            r"^uptime$",
            r"^w$",
            r"^who$",
            r"^id$",
            r"^uname(\s|$)"
        ],
        PermissionLevel.LOW_RISK: [
            r"^systemctl\s+status",
            r"^journalctl(\s|$)",
            r"^docker\s+ps",
            r"^docker\s+logs",
            r"^kubectl\s+get",
            r"^kubectl\s+describe",
            r"^git\s+status",
            r"^git\s+log",
            r"^npm\s+list",
            r"^pip\s+list",
            r"^curl\s+-I",  # HEADリクエストのみ
            r"^wget\s+--spider"  # ダウンロードなし
        ],
        PermissionLevel.MEDIUM_RISK: [
            r"^systemctl\s+restart\s+.*_dev",
            r"^docker\s+restart\s+.*_dev",
            r"^kubectl\s+rollout\s+restart.*staging",
            r"^git\s+pull",
            r"^npm\s+install",
            r"^pip\s+install",
            r"^apt-get\s+update$",
            r"^yum\s+check-update$"
        ],
        PermissionLevel.HIGH_RISK: [
            r"^systemctl\s+restart\s+(?!.*critical)",
            r"^docker\s+restart\s+(?!.*critical)",
            r"^kubectl\s+scale",
            r"^kubectl\s+rollout",
            r"^nginx\s+-s\s+reload"
        ],
        PermissionLevel.CRITICAL: []  # 全て手動承認
    }
    
    # 禁止コマンドパターン
    FORBIDDEN_PATTERNS = [
        r"rm\s+-rf\s+/",
        r"dd\s+.*of=/dev/",
        r"mkfs",
        r"format",
        r"fdisk",
        r"shutdown",
        r"reboot",
        r"init\s+0",
        r":(){ :|:& };:",  # Fork bomb
        r">\s*/dev/sda",
        r"wget.*\|\s*sh",  # Remote script execution
        r"curl.*\|\s*sh"
    ]
    
    @classmethod
    def validate_command(cls, command: str, permission_level: PermissionLevel) -> Tuple[bool, Optional[str]]:
        """コマンドの妥当性検証"""
        command = command.strip()
        
        # 空コマンドチェック
        if not command:
            return False, "Empty command"
        
        # 禁止パターンチェック
        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Forbidden command pattern detected: {pattern}"
        
        # 権限レベルに応じたチェック
        if permission_level == PermissionLevel.CRITICAL:
            # CRITICALは全て手動承認
            return True, None
        
        # 許可パターンチェック
        allowed_patterns = []
        level_hierarchy = [
            PermissionLevel.READ_ONLY,
            PermissionLevel.LOW_RISK,
            PermissionLevel.MEDIUM_RISK,
            PermissionLevel.HIGH_RISK,
            PermissionLevel.CRITICAL
        ]
        
        current_level_index = level_hierarchy.index(permission_level)
        for i in range(current_level_index + 1):
            level = level_hierarchy[i]
            allowed_patterns.extend(cls.SAFE_COMMANDS.get(level, []))
        
        for pattern in allowed_patterns:
            if re.match(pattern, command):
                return True, None
        
        return False, f"Command not allowed for permission level {permission_level.value}"
    
    @classmethod
    def sanitize_command(cls, command: str) -> str:
        """コマンドのサニタイズ"""
        # 危険な文字をエスケープ
        dangerous_chars = [';', '&', '|', '`', '$', '(', ')', '<', '>', '\n']
        sanitized = command
        
        for char in dangerous_chars:
            if char in sanitized and not (char == '|' and 'grep' in sanitized):
                # grepのパイプは許可
                sanitized = sanitized.replace(char, f'\\{char}')
        
        return sanitized


class SSHConnectionPool:
    """SSH接続プール"""
    
    def __init__(self, max_connections: int = 10, connection_timeout: int = 30):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self._connections: Dict[str, SSHClientConnection] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_used: Dict[str, datetime] = {}
        self._cleanup_interval = 300  # 5分
        self._cleanup_task = None
    
    async def start(self):
        """接続プール開始"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SSH connection pool started", max_connections=self.max_connections)
    
    async def stop(self):
        """接続プール停止"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # 全接続をクローズ
        for conn_id in list(self._connections.keys()):
            await self._close_connection(conn_id)
        
        logger.info("SSH connection pool stopped")
    
    async def _cleanup_loop(self):
        """定期的な接続クリーンアップ"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Connection cleanup error", error=str(e))
    
    async def _cleanup_stale_connections(self):
        """古い接続をクリーンアップ"""
        now = datetime.now()
        stale_threshold = timedelta(minutes=15)
        
        for conn_id in list(self._connections.keys()):
            if conn_id in self._last_used:
                if now - self._last_used[conn_id] > stale_threshold:
                    await self._close_connection(conn_id)
                    logger.debug("Cleaned up stale connection", connection_id=conn_id)
    
    def _get_connection_id(self, target: SSHTarget) -> str:
        """接続IDを生成"""
        return f"{target.username}@{target.host}:{target.port}"
    
    async def get_connection(self, target: SSHTarget) -> SSHClientConnection:
        """接続を取得"""
        conn_id = self._get_connection_id(target)
        
        # ロックを取得または作成
        if conn_id not in self._locks:
            self._locks[conn_id] = asyncio.Lock()
        
        async with self._locks[conn_id]:
            # 既存接続チェック
            if conn_id in self._connections:
                conn = self._connections[conn_id]
                # 接続が生きているかチェック
                try:
                    await conn.run("echo alive", timeout=5)
                    self._last_used[conn_id] = datetime.now()
                    return conn
                except:
                    # 接続が死んでいる場合は削除
                    await self._close_connection(conn_id)
            
            # 新規接続作成
            conn = await self._create_connection(target)
            self._connections[conn_id] = conn
            self._last_used[conn_id] = datetime.now()
            
            # 接続数制限チェック
            if len(self._connections) > self.max_connections:
                # 最も古い接続を削除
                oldest_id = min(self._last_used.keys(), key=lambda k: self._last_used[k])
                await self._close_connection(oldest_id)
            
            return conn
    
    async def _create_connection(self, target: SSHTarget) -> SSHClientConnection:
        """新規SSH接続作成"""
        connect_kwargs = {
            "host": target.host,
            "port": target.port,
            "username": target.username,
            "known_hosts": None,  # ホスト検証を無効化（本番では要設定）
            "connect_timeout": self.connection_timeout
        }
        
        if target.key_file:
            connect_kwargs["client_keys"] = [target.key_file]
        elif target.password:
            connect_kwargs["password"] = target.password
        
        conn = await asyncssh.connect(**connect_kwargs)
        logger.info("SSH connection established", 
                   host=target.host, 
                   username=target.username,
                   environment=target.environment)
        return conn
    
    async def _close_connection(self, conn_id: str):
        """接続をクローズ"""
        if conn_id in self._connections:
            try:
                conn = self._connections[conn_id]
                conn.close()
                await conn.wait_closed()
            except:
                pass
            
            del self._connections[conn_id]
            if conn_id in self._last_used:
                del self._last_used[conn_id]
            
            logger.debug("SSH connection closed", connection_id=conn_id)


class SSHExecutor:
    """SSH実行エンジン"""
    
    def __init__(self):
        """初期化"""
        self.config_manager = get_config_manager()
        self.config = self.config_manager.get_module_config("ssh_executor")
        
        # 接続プール
        pool_size = self.config.config.get("connection_pool_size", 10)
        self.connection_pool = SSHConnectionPool(max_connections=pool_size)
        
        # 承認設定
        self.approval_required = self.config.config.get("approval_required", {})
        self.auto_approve_patterns = self.config.config.get("auto_approve_patterns", {})
        self.approval_timeout_minutes = self.config.config.get("approval_timeout_minutes", 30)
        
        # 実行制限
        self.max_parallel_executions = self.config.config.get("max_parallel_executions", 5)
        self.default_timeout = self.config.config.get("default_timeout_seconds", 300)
        
        # 承認待ちキュー
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        
        # 実行中のタスク
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("SSH Executor initialized",
                   pool_size=pool_size,
                   max_parallel=self.max_parallel_executions)
    
    async def start(self):
        """SSH Executor開始"""
        await self.connection_pool.start()
        logger.info("SSH Executor started")
    
    async def stop(self):
        """SSH Executor停止"""
        # 実行中のタスクをキャンセル
        for task_id, task in self.running_tasks.items():
            task.cancel()
        
        await self.connection_pool.stop()
        logger.info("SSH Executor stopped")
    
    async def verify_setup(self) -> bool:
        """セットアップ確認"""
        try:
            # 設定確認
            if not self.config.enabled:
                logger.warning("SSH Executor is disabled in configuration")
                return False
            
            # SSH鍵ファイル存在確認
            default_key = self.config.config.get("default_key_file")
            if default_key and not Path(default_key).exists():
                logger.warning("Default SSH key file not found", key_file=default_key)
                return False
            
            logger.info("SSH Executor setup verified successfully")
            return True
            
        except Exception as e:
            logger.error("SSH Executor setup verification failed",
                        error=str(e), exception_type=type(e).__name__)
            return False
    
    def create_approval_request(self, command_request: CommandRequest) -> ApprovalRequest:
        """承認リクエスト作成"""
        # 権限レベルに応じた承認者決定
        required_approvers = self._get_required_approvers(command_request)
        
        # 有効期限設定
        expiry_time = datetime.now() + timedelta(minutes=self.approval_timeout_minutes)
        
        approval_request = ApprovalRequest(
            command_request=command_request,
            required_approvers=required_approvers,
            expiry_time=expiry_time
        )
        
        # 自動承認チェック
        if self._check_auto_approval(command_request):
            approval_request.approval_status = ApprovalStatus.AUTO_APPROVED
            logger.info("Command auto-approved",
                       request_id=command_request.request_id,
                       command=command_request.command[:50])
        
        return approval_request
    
    def _get_required_approvers(self, command_request: CommandRequest) -> List[str]:
        """必要な承認者リストを取得"""
        permission_level = command_request.permission_level
        environment = command_request.targets[0].environment if command_request.targets else "development"
        
        # 設定から承認者を取得
        approvers_config = self.approval_required.get(permission_level.value, {})
        
        if isinstance(approvers_config, dict):
            return approvers_config.get(environment, [])
        elif isinstance(approvers_config, list):
            return approvers_config
        else:
            return []
    
    def _check_auto_approval(self, command_request: CommandRequest) -> bool:
        """自動承認可能かチェック"""
        # READ_ONLYは常に自動承認
        if command_request.permission_level == PermissionLevel.READ_ONLY:
            return True
        
        # dry-runは自動承認
        if command_request.dry_run:
            return True
        
        # パターンマッチング
        patterns = self.auto_approve_patterns.get(command_request.permission_level.value, [])
        for pattern in patterns:
            if re.match(pattern, command_request.command):
                return True
        
        return False
    
    async def submit_request(self, command_request: CommandRequest) -> ApprovalRequest:
        """実行リクエスト送信"""
        # コマンドバリデーション
        is_valid, error_msg = CommandValidator.validate_command(
            command_request.command, 
            command_request.permission_level
        )
        
        if not is_valid:
            raise ValueError(f"Invalid command: {error_msg}")
        
        # 承認リクエスト作成
        approval_request = self.create_approval_request(command_request)
        
        # 承認待ちキューに追加
        self.pending_approvals[approval_request.approval_id] = approval_request
        
        # 監査ログ
        audit_log(AuditLogEntry(
            action=AuditAction.CREATE,
            resource=f"ssh_command_request:{command_request.request_id}",
            result="success",
            details=f"Command: {command_request.command[:100]}",
            user_id=command_request.requested_by,
            risk_level=self._get_risk_level(command_request.permission_level)
        ))
        
        logger.info("Command request submitted",
                   request_id=command_request.request_id,
                   approval_id=approval_request.approval_id,
                   auto_approved=approval_request.approval_status == ApprovalStatus.AUTO_APPROVED)
        
        # 自動承認の場合は即実行
        if approval_request.is_approved():
            asyncio.create_task(self._execute_approved_request(approval_request))
        
        return approval_request
    
    def _get_risk_level(self, permission_level: PermissionLevel) -> str:
        """リスクレベル変換"""
        risk_map = {
            PermissionLevel.READ_ONLY: "low",
            PermissionLevel.LOW_RISK: "low",
            PermissionLevel.MEDIUM_RISK: "medium",
            PermissionLevel.HIGH_RISK: "high",
            PermissionLevel.CRITICAL: "critical"
        }
        return risk_map.get(permission_level, "medium")
    
    async def approve_request(self, approval_id: str, approver: str, decision: bool = True) -> bool:
        """リクエスト承認"""
        if approval_id not in self.pending_approvals:
            logger.warning("Approval request not found", approval_id=approval_id)
            return False
        
        approval_request = self.pending_approvals[approval_id]
        
        # 期限チェック
        if approval_request.is_expired():
            approval_request.approval_status = ApprovalStatus.EXPIRED
            logger.warning("Approval request expired", approval_id=approval_id)
            return False
        
        # 承認記録
        approval_request.approvals[approver] = {
            "decision": decision,
            "timestamp": datetime.now().isoformat()
        }
        
        if not decision:
            approval_request.approval_status = ApprovalStatus.REJECTED
            logger.info("Command request rejected",
                       approval_id=approval_id,
                       approver=approver)
            return True
        
        # 全承認者の承認が揃ったかチェック
        if len(approval_request.approvals) >= len(approval_request.required_approvers):
            approval_request.approval_status = ApprovalStatus.APPROVED
            logger.info("Command request fully approved",
                       approval_id=approval_id,
                       approvers=list(approval_request.approvals.keys()))
            
            # 実行開始
            asyncio.create_task(self._execute_approved_request(approval_request))
        
        return True
    
    async def _execute_approved_request(self, approval_request: ApprovalRequest):
        """承認済みリクエストを実行"""
        command_request = approval_request.command_request
        
        # 実行タスク作成
        if command_request.parallel_execution and len(command_request.targets) > 1:
            # 並列実行
            results = await self._execute_parallel(command_request)
        else:
            # 順次実行
            results = await self._execute_sequential(command_request)
        
        # 実行レポート作成
        overall_status = self._determine_overall_status(results)
        total_time = sum(r.execution_time for r in results)
        
        report = ExecutionReport(
            request_id=command_request.request_id,
            command_request=command_request,
            approval_request=approval_request,
            results=results,
            overall_status=overall_status,
            total_execution_time=total_time
        )
        
        # 監査ログ
        audit_log(AuditLogEntry(
            action=AuditAction.EXECUTE,
            resource=f"ssh_command:{command_request.request_id}",
            result="success" if overall_status == ExecutionStatus.COMPLETED else "failure",
            details=f"Executed on {len(results)} targets, success rate: {report.to_dict()['success_rate']:.1%}",
            user_id=command_request.requested_by,
            risk_level=self._get_risk_level(command_request.permission_level)
        ))
        
        # 承認待ちキューから削除
        if approval_request.approval_id in self.pending_approvals:
            del self.pending_approvals[approval_request.approval_id]
        
        logger.info("Command execution completed",
                   request_id=command_request.request_id,
                   overall_status=overall_status.value,
                   total_time=f"{total_time:.2f}s")
        
        return report
    
    async def _execute_parallel(self, command_request: CommandRequest) -> List[CommandResult]:
        """並列実行"""
        tasks = []
        semaphore = asyncio.Semaphore(self.max_parallel_executions)
        
        for target in command_request.targets:
            task = self._execute_with_semaphore(semaphore, command_request, target)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 例外を結果に変換
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(self._create_error_result(
                    command_request, 
                    command_request.targets[i],
                    str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def _execute_sequential(self, command_request: CommandRequest) -> List[CommandResult]:
        """順次実行"""
        results = []
        
        for target in command_request.targets:
            try:
                result = await self._execute_single(command_request, target)
                results.append(result)
                
                # エラーで中断するかチェック
                if result.exit_code != 0 and not command_request.context.get("continue_on_error", False):
                    logger.warning("Stopping sequential execution due to error",
                                  target=target.host,
                                  exit_code=result.exit_code)
                    break
                    
            except Exception as e:
                results.append(self._create_error_result(command_request, target, str(e)))
                if not command_request.context.get("continue_on_error", False):
                    break
        
        return results
    
    async def _execute_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        command_request: CommandRequest, 
        target: SSHTarget
    ) -> CommandResult:
        """セマフォ付き実行"""
        async with semaphore:
            return await self._execute_single(command_request, target)
    
    async def _execute_single(self, command_request: CommandRequest, target: SSHTarget) -> CommandResult:
        """単一ターゲットでコマンド実行"""
        start_time = time.time()
        
        logger.info("Executing command on target",
                   request_id=command_request.request_id,
                   target=target.host,
                   command=command_request.command[:50])
        
        try:
            # dry-runチェック
            if command_request.dry_run:
                return CommandResult(
                    request_id=command_request.request_id,
                    target=target,
                    command=command_request.command,
                    exit_code=0,
                    stdout=f"[DRY-RUN] Command would be executed: {command_request.command}",
                    stderr="",
                    execution_time=0.0,
                    status=ExecutionStatus.COMPLETED,
                    completed_at=datetime.now()
                )
            
            # SSH接続取得
            conn = await self.connection_pool.get_connection(target)
            
            # コマンド実行
            result = await asyncio.wait_for(
                conn.run(command_request.command),
                timeout=command_request.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            return CommandResult(
                request_id=command_request.request_id,
                target=target,
                command=command_request.command,
                exit_code=result.returncode,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                execution_time=execution_time,
                status=ExecutionStatus.COMPLETED if result.returncode == 0 else ExecutionStatus.FAILED,
                completed_at=datetime.now()
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error("Command execution timeout",
                        request_id=command_request.request_id,
                        target=target.host,
                        timeout=command_request.timeout_seconds)
            
            return CommandResult(
                request_id=command_request.request_id,
                target=target,
                command=command_request.command,
                exit_code=-1,
                stdout="",
                stderr=f"Command execution timeout after {command_request.timeout_seconds} seconds",
                execution_time=execution_time,
                status=ExecutionStatus.TIMEOUT,
                error_message="Timeout",
                completed_at=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Command execution failed",
                        request_id=command_request.request_id,
                        target=target.host,
                        error=str(e),
                        exception_type=type(e).__name__)
            
            return CommandResult(
                request_id=command_request.request_id,
                target=target,
                command=command_request.command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                completed_at=datetime.now()
            )
    
    def _create_error_result(self, command_request: CommandRequest, target: SSHTarget, error: str) -> CommandResult:
        """エラー結果作成"""
        return CommandResult(
            request_id=command_request.request_id,
            target=target,
            command=command_request.command,
            exit_code=-1,
            stdout="",
            stderr=error,
            execution_time=0.0,
            status=ExecutionStatus.FAILED,
            error_message=error,
            completed_at=datetime.now()
        )
    
    def _determine_overall_status(self, results: List[CommandResult]) -> ExecutionStatus:
        """全体ステータス判定"""
        if not results:
            return ExecutionStatus.FAILED
        
        statuses = [r.status for r in results]
        
        if all(s == ExecutionStatus.COMPLETED for s in statuses):
            return ExecutionStatus.COMPLETED
        elif any(s == ExecutionStatus.TIMEOUT for s in statuses):
            return ExecutionStatus.TIMEOUT
        elif any(s == ExecutionStatus.CANCELLED for s in statuses):
            return ExecutionStatus.CANCELLED
        else:
            return ExecutionStatus.FAILED
    
    async def execute_command(
        self,
        command: str,
        targets: List[SSHTarget],
        permission_level: PermissionLevel = PermissionLevel.READ_ONLY,
        timeout: int = None,
        dry_run: bool = False,
        context: Dict[str, Any] = None,
        requested_by: str = "system",
        correlation_id: Optional[str] = None
    ) -> ExecutionReport:
        """コマンド実行（高レベルAPI）"""
        
        # リクエスト作成
        command_request = CommandRequest(
            command=command,
            targets=targets,
            permission_level=permission_level,
            timeout_seconds=timeout or self.default_timeout,
            dry_run=dry_run,
            context=context or {},
            requested_by=requested_by,
            correlation_id=correlation_id
        )
        
        # 承認リクエスト送信
        approval_request = await self.submit_request(command_request)
        
        # 自動承認の場合は完了まで待機
        if approval_request.is_approved():
            # 実行完了を待つ（簡易的な実装）
            await asyncio.sleep(0.1)  # 実行開始を待つ
            
            # TODO: より洗練された完了待機メカニズムが必要
            max_wait = command_request.timeout_seconds + 10
            start_time = time.time()
            
            while approval_request.approval_id in self.pending_approvals and time.time() - start_time < max_wait:
                await asyncio.sleep(1)
            
            # ダミーレポート返却（実際は実行結果を追跡する必要がある）
            return ExecutionReport(
                request_id=command_request.request_id,
                command_request=command_request,
                approval_request=approval_request,
                results=[],
                overall_status=ExecutionStatus.COMPLETED,
                total_execution_time=0.0
            )
        else:
            # 承認待ち
            return ExecutionReport(
                request_id=command_request.request_id,
                command_request=command_request,
                approval_request=approval_request,
                results=[],
                overall_status=ExecutionStatus.QUEUED,
                total_execution_time=0.0
            )
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """承認待ちリスト取得"""
        # 期限切れを更新
        for approval_id, approval in list(self.pending_approvals.items()):
            if approval.is_expired():
                approval.approval_status = ApprovalStatus.EXPIRED
                del self.pending_approvals[approval_id]
        
        return list(self.pending_approvals.values())
    
    def get_approval_status(self, approval_id: str) -> Optional[ApprovalRequest]:
        """承認ステータス取得"""
        return self.pending_approvals.get(approval_id)


# モジュール使用例
async def main():
    """使用例"""
    executor = SSHExecutor()
    
    # 開始
    await executor.start()
    
    try:
        # セットアップ確認
        if not await executor.verify_setup():
            print("❌ Setup verification failed")
            return
        
        # テストターゲット
        targets = [
            SSHTarget(
                host="localhost",
                username="test",
                environment="development"
            )
        ]
        
        # 読み取り専用コマンド実行
        report = await executor.execute_command(
            command="ls -la /tmp",
            targets=targets,
            permission_level=PermissionLevel.READ_ONLY,
            dry_run=True,
            requested_by="demo_user"
        )
        
        print(f"📊 Execution Report")
        print(f"   Request ID: {report.request_id}")
        print(f"   Status: {report.overall_status.value}")
        print(f"   Approval: {report.approval_request.approval_status.value if report.approval_request else 'N/A'}")
        
        # 承認待ちリスト
        pending = executor.get_pending_approvals()
        print(f"\n⏳ Pending Approvals: {len(pending)}")
        
    finally:
        # 停止
        await executor.stop()


if __name__ == "__main__":
    asyncio.run(main())
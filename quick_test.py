#!/usr/bin/env python3
"""
AALS クイックテスト
各モジュールの主要機能を素早く確認
"""

import asyncio
from aals.core.config import get_config
from aals.core.logger import get_logger, AuditAction, audit_log, AuditLogEntry


async def quick_test():
    """クイックテスト実行"""
    print("⚡ AALS クイックテスト")
    print("=" * 40)
    
    # 1. 設定確認
    config = get_config()
    print(f"🔧 環境: {config.environment}")
    print(f"🔧 デバッグ: {config.debug}")
    
    # 2. ログテスト
    logger = get_logger("quick_test")
    logger.info("クイックテスト開始", test_type="quick")
    print("📝 ログ出力: 正常")
    
    # 3. 監査ログテスト
    audit_log(AuditLogEntry(
        action=AuditAction.EXECUTE,
        resource="quick_test",
        result="success",
        details="クイックテスト実行"
    ))
    print("🔒 監査ログ: 正常")
    
    # 4. 設定別モジュール確認
    from aals.core.config import get_config_manager
    manager = get_config_manager()
    
    modules = ["config_manager", "slack_alert_reader", "basic_logger"]
    for module in modules:
        status = "✅" if manager.is_module_enabled(module) else "❌"
        print(f"{status} {module}")
    
    print("\n🎉 クイックテスト完了!")


if __name__ == "__main__":
    asyncio.run(quick_test())
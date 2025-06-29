# AALS - Agentic AI Auto Learning System

SRE・インフラエンジニア向けの自己学習型AIアシスタントシステム

## 概要

AALSは、人間と同じように学習・記憶し、知見を蓄積してタスクや作業を増やしていく自己学習型AIシステムです。特にSRE（Site Reliability Engineering）とインフラエンジニアリングの分野に特化して設計されています。

## 特徴

- **段階的学習**: 人間のように経験から学び、継続的に能力を向上
- **マイクロモジュール設計**: 独立したモジュールによる段階的な機能拡張
- **安全性重視**: 本番環境での安全な動作を保証する多層承認システム
- **ツール統合**: Slack、GitHub、Prometheus、SSH等の既存ツールとの完全統合

## システム構成

### Phase 1: 基盤モジュール群
- **Module 1**: Config Manager - 設定管理とバリデーション
- **Module 2**: Slack Alert Reader - Slackアラート監視
- **Module 3**: Basic Logger - ログ記録と監査証跡

### Phase 2: 分析モジュール群
- **Module 4**: Prometheus Analyzer - メトリクス分析
- **Module 5**: GitHub Issues Searcher - インシデント履歴検索
- **Module 6**: LLM Wrapper - AI推論エンジン

### Phase 3: 統合・自動化モジュール
- **Module 7**: Alert Correlator - アラート相関分析
- **Module 8**: SSH Executor - リモートコマンド実行
- **Module 9**: Response Orchestrator - ワークフロー統合

## 技術スタック

- **Python 3.11+**: メインプログラミング言語
- **FastAPI**: APIサーバーフレームワーク
- **PostgreSQL + pgvector**: データベースとベクトル検索
- **Redis**: キャッシュとセッション管理
- **Docker**: コンテナ化
- **Pydantic**: 設定管理とデータバリデーション

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/your-org/aals.git
cd aals

# 仮想環境を作成
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 依存関係をインストール
pip install -e .

# 設定ファイルを準備
cp .env.example .env
# .envファイルを編集して必要な設定を行う
```

## 設定

### 環境変数

重要な設定は環境変数で管理されます：

```bash
# 必須設定
AALS_SECRET_KEY=your-secret-key-here
AALS_ENVIRONMENT=development

# データベース設定
AALS_DB_PASSWORD=your-postgres-password
AALS_REDIS_PASSWORD=your-redis-password

# 統合サービス設定
AALS_SLACK_TOKEN=xoxb-your-slack-bot-token
AALS_GITHUB_TOKEN=ghp_your-github-token
AALS_CLAUDE_API_KEY=sk-ant-your-claude-key
```

### 設定ファイル

YAMLファイルによる詳細設定：

```yaml
# config/default.yaml
modules:
  slack_alert_reader:
    enabled: true
    channels: ["#alerts", "#incidents"]
    
  prometheus_analyzer:
    enabled: true
    endpoints: ["http://prometheus:9090"]
```

## 使用方法

### 基本的な使用例

```python
from aals.core.config import get_config

# 設定を取得
config = get_config()

# モジュールの有効化チェック
if config.is_module_enabled("slack_alert_reader"):
    # Slackアラート監視を開始
    pass
```

### テスト実行

```bash
# 全テストを実行
pytest

# カバレッジ付きテスト
pytest --cov=aals --cov-report=html

# 特定のモジュールをテスト
pytest tests/test_config.py
```

## 開発

### 開発環境のセットアップ

```bash
# 開発用依存関係をインストール
pip install -e ".[dev]"

# pre-commitフックを設定
pre-commit install

# コードフォーマット
black aals/
isort aals/

# 型チェック
mypy aals/
```

### モジュール開発

各モジュールは独立して開発・テスト可能です：

1. 2-3日で完全動作するモジュールを実装
2. 独立したテストスイートを作成
3. 設定ファイルによる有効化/無効化対応
4. 他モジュールとの統合

## ライセンス

MIT License

## 貢献

Issues、Pull Requestを歓迎します。

## 開発状況

### 完了済み
- [x] プロジェクト基盤構築
- [x] Module 1: Config Manager

### 進行中
- [ ] Module 1 テスト・動作確認

### 予定
- [ ] Module 2: Slack Alert Reader
- [ ] Module 3: Basic Logger
- [ ] その他モジュール群

---

> **注意**: このシステムは本番環境での使用を想定しています。適切なセキュリティ設定と承認フローを確認してから運用してください。
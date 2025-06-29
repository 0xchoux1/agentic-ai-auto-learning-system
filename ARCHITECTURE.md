# AALS システムアーキテクチャ

## 概要

AALS（Agentic AI Auto Learning System）は、SRE・インフラエンジニア向けの自己学習型AIアシスタントシステムです。マイクロモジュール設計により、段階的な機能拡張と独立した動作を実現しています。

## 設計方針

### 1. マイクロモジュール設計
- **独立性**: 各モジュールは他への依存を最小限に抑制
- **段階的導入**: 2-3日で完成する小単位モジュール
- **設定駆動**: YAML設定による有効化/無効化制御

### 2. 安全性重視
- **段階的権限**: READ-ONLY → 承認制 → 限定自動化
- **環境別制御**: 本番/ステージング/開発環境の厳格分離
- **監査証跡**: 全操作の完全なログ記録

### 3. LLMコンテキスト制限対応
- **自己完結モジュール**: 複雑な依存関係を回避
- **ドキュメント自動更新**: 各セッションでの進捗記録
- **段階的統合**: 順次結合による品質保証

## システム構成

```
aals/
├── core/                    # 共通基盤機能
│   ├── config.py           # 設定管理（Module 1）
│   ├── logger.py           # ログ記録（Module 3）
│   └── security.py         # セキュリティ機能
├── modules/                 # 独立機能モジュール
│   ├── slack_reader.py     # Slackアラート読取（Module 2）
│   ├── prometheus_analyzer.py # メトリクス分析（Module 4）
│   ├── github_searcher.py  # GitHub検索（Module 5）
│   ├── llm_wrapper.py      # LLM統合（Module 6）
│   ├── alert_correlator.py # アラート相関（Module 7）
│   ├── ssh_executor.py     # SSH実行（Module 8）
│   └── orchestrator.py     # 統合制御（Module 9）
├── integrations/           # 外部サービス統合
│   ├── slack_client.py     # Slack API
│   ├── github_client.py    # GitHub API
│   ├── prometheus_client.py # Prometheus API
│   └── ssh_client.py       # SSH接続
└── tests/                  # テストスイート
    ├── test_config.py      # 設定管理テスト
    └── fixtures/           # テストデータ
```

## モジュール詳細

### Phase 1: 基盤モジュール群

#### Module 1: Config Manager ✅ **完成**
**目的**: 全システムの設定管理基盤
**機能**:
- YAML設定ファイル + 環境変数統合
- 型安全な設定オブジェクト（Pydantic使用）
- 環境別設定バリデーション
- モジュール有効化制御

**技術**:
- `pydantic-settings`: 設定管理
- `PyYAML`: YAML解析
- `python-dotenv`: 環境変数

**使用例**:
```python
from aals.core.config import get_config
config = get_config()
if config.is_module_enabled("slack_alert_reader"):
    # モジュール実行
```

#### Module 2: Slack Alert Reader 🔄 **次の実装対象**
**目的**: Slackからアラート情報を取得・分析
**機能**:
- Slack API統合
- アラート履歴の構造化
- 基本パターン認識

#### Module 3: Basic Logger 📋 **予定**
**目的**: 統一ログ記録・監査証跡
**機能**:
- 構造化ログ出力
- 監査ログ記録
- ローテーション機能

### Phase 2: 分析モジュール群

#### Module 4: Prometheus Analyzer 📋 **予定**
**目的**: メトリクス取得・異常検知
**機能**:
- Prometheus API統合
- 時系列データ分析
- 異常値検出

#### Module 5: GitHub Issues Searcher 📋 **予定**
**目的**: 過去インシデント検索・分析
**機能**:
- GitHub Issues API統合
- 類似ケース検索
- ナレッジ抽出

#### Module 6: LLM Wrapper 📋 **予定**
**目的**: AI推論エンジン統合
**機能**:
- Claude/OpenAI API統合
- コスト管理・使用量制御
- プロンプトテンプレート管理

### Phase 3: 統合・自動化モジュール

#### Module 7: Alert Correlator 📋 **予定**
**目的**: アラート相関分析・推奨アクション生成
**統合対象**: Module 2, 4, 5

#### Module 8: SSH Executor 📋 **予定**
**目的**: リモートコマンド安全実行
**機能**:
- SSH接続管理
- 承認ワークフロー
- 実行ログ記録

#### Module 9: Response Orchestrator 📋 **予定**
**目的**: 全モジュール統合制御
**機能**:
- ワークフロー管理
- 権限制御
- 人間承認インターフェース

## データフロー

```
Slack Alert → Module 2 → Module 7 → Module 9
              ↓
Prometheus → Module 4 → ↑
              ↓
GitHub → Module 5 → ↑
              ↓
LLM → Module 6 → ↑
              ↓
SSH → Module 8 → ↑
```

## セキュリティアーキテクチャ

### 権限レベル
1. **READ_ONLY**: データ参照のみ（自動実行可）
2. **LOW_RISK**: 開発環境変更（事後報告）
3. **MEDIUM_RISK**: ステージング変更（事前承認）
4. **HIGH_RISK**: 本番環境変更（複数人承認）
5. **CRITICAL**: 手動実行必須

### 環境分離
- **Development**: 全機能有効、学習データ蓄積
- **Staging**: 制限付き自動化、テスト環境
- **Production**: 最小権限、承認制ワークフロー

## 技術スタック

### コア技術
- **Python 3.11+**: メインプログラミング言語
- **FastAPI**: API サーバー フレームワーク
- **Pydantic**: データバリデーション・設定管理
- **PostgreSQL + pgvector**: データ永続化・ベクトル検索
- **Redis**: キャッシュ・セッション管理

### 統合技術
- **Slack SDK**: Slack API統合
- **PyGithub**: GitHub API統合
- **prometheus-client**: Prometheus統合
- **paramiko/asyncssh**: SSH接続
- **OpenAI/Anthropic SDK**: LLM統合

### 開発・テスト
- **pytest**: テストフレームワーク
- **Docker**: コンテナ化
- **black/isort**: コードフォーマット
- **mypy**: 静的型チェック

## パフォーマンス設計

### レスポンス時間目標
- **緊急時対応**: 5秒以内（キャッシュ活用）
- **通常分析**: 30秒以内（LLM使用）
- **詳細調査**: 5分以内（包括的分析）

### スケーラビリティ
- **水平スケーリング**: モジュール別コンテナ展開
- **非同期処理**: FastAPI + asyncio
- **キャッシュ戦略**: Redis + 階層キャッシュ

## 拡張性

### 新モジュール追加
1. `aals/modules/new_module.py` 作成
2. 設定ファイルにモジュール定義追加
3. テストスイート作成
4. 段階的統合テスト

### 外部ツール統合
1. `aals/integrations/tool_client.py` 作成
2. 認証・API設定追加
3. エラーハンドリング実装
4. 統合テスト作成

---

> このアーキテクチャは、実装進捗に応じて継続的に更新されます。各モジュール完成時に詳細仕様を追記します。
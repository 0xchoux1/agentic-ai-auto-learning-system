# AALS Deployment Guide

本番環境でのAALS（Agentic AI Auto Learning System）デプロイメントガイド

## 🚀 クイックスタート

### 前提条件

- Docker 20.10+
- Docker Compose 2.0+
- 最低 4GB RAM
- 最低 10GB ディスク容量

### 1. デプロイメント実行

```bash
# リポジトリをクローン
git clone https://github.com/your-org/aals.git
cd aals

# デプロイスクリプトを実行
./scripts/deploy.sh
```

初回実行時は`.env`ファイルが作成されるので、必要な設定を行ってから再実行してください。

### 2. 必須設定項目

`.env`ファイルで以下の項目を設定：

```bash
# セキュリティ設定
AALS_SECRET_KEY=your-super-secret-key-here
AALS_DB_PASSWORD=your-secure-database-password
AALS_REDIS_PASSWORD=your-secure-redis-password

# API統合設定
AALS_SLACK_TOKEN=xoxb-your-slack-bot-token
AALS_GITHUB_TOKEN=ghp_your-github-token
AALS_CLAUDE_API_KEY=sk-ant-your-claude-key
```

### 3. 動作確認

```bash
# ヘルスチェック
curl http://localhost:8000/health

# サービス状態確認
docker-compose ps

# ログ確認
docker-compose logs -f aals
```

## 📋 詳細設定

### アーキテクチャ概要

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AALS App      │    │   PostgreSQL    │    │     Redis       │
│   Port: 8000    │────│   Port: 5432    │    │   Port: 6379    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         │
┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │    Grafana      │
│   Port: 9090    │    │   Port: 3000    │
└─────────────────┘    └─────────────────┘
```

### サービス一覧

| サービス | ポート | 説明 | ヘルスチェック |
|---------|--------|------|----------------|
| AALS App | 8000 | メインアプリケーション | `/health` |
| PostgreSQL | 5432 | データベース | `pg_isready` |
| Redis | 6379 | キャッシュ・セッション | `ping` |
| Prometheus | 9090 | メトリクス収集 | `/` |
| Grafana | 3000 | 可視化ダッシュボード | `/api/health` |

## 🔧 設定詳細

### 環境変数一覧

#### コア設定
```bash
AALS_ENVIRONMENT=production        # 環境設定
AALS_SECRET_KEY=<secret>          # JWT署名用秘密鍵
AALS_DEBUG=false                  # デバッグモード
AALS_LOG_LEVEL=INFO              # ログレベル
```

#### データベース設定
```bash
AALS_DB_HOST=postgres            # DBホスト
AALS_DB_PORT=5432               # DBポート
AALS_DB_NAME=aals               # DB名
AALS_DB_USER=aals_user          # DBユーザー
AALS_DB_PASSWORD=<password>     # DBパスワード
```

#### Redis設定
```bash
AALS_REDIS_HOST=redis           # Redisホスト
AALS_REDIS_PORT=6379           # Redisポート
AALS_REDIS_PASSWORD=<password>  # Redisパスワード
```

#### API統合設定
```bash
AALS_SLACK_TOKEN=<token>        # Slack Bot Token
AALS_GITHUB_TOKEN=<token>       # GitHub Personal Access Token
AALS_CLAUDE_API_KEY=<key>       # Claude API Key
AALS_PROMETHEUS_URL=<url>       # Prometheus URL
```

### SSH実行設定

```bash
AALS_SSH_KEY_PATH=/app/data/ssh_keys/id_rsa
AALS_SSH_KNOWN_HOSTS_PATH=/app/data/ssh_keys/known_hosts
AALS_SSH_ALLOWED_HOSTS=server1.example.com,server2.example.com
AALS_SSH_REQUIRED_APPROVERS=admin@example.com,sre@example.com
```

## 🔒 セキュリティ設定

### 1. SSH鍵設定

```bash
# SSH鍵を生成
ssh-keygen -t rsa -b 4096 -f ./data/ssh_keys/id_rsa

# 公開鍵を対象サーバーに配布
ssh-copy-id -i ./data/ssh_keys/id_rsa.pub user@target-server
```

### 2. ファイアウォール設定

```bash
# 必要なポートのみ開放
ufw allow 8000    # AALS API
ufw allow 3000    # Grafana (必要に応じて)
ufw allow 9090    # Prometheus (必要に応じて)
```

### 3. SSL/TLS設定

プロダクション環境では、リバースプロキシ（Nginx等）でSSL終端を実装：

```nginx
server {
    listen 443 ssl;
    server_name aals.yourdomain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 監視・運用

### ログ管理

```bash
# アプリケーションログ
docker-compose logs -f aals

# 監査ログ
docker-compose exec aals tail -f /app/logs/audit.log

# データベースログ
docker-compose logs postgres
```

### メトリクス監視

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### バックアップ

```bash
# データベースバックアップ
docker-compose exec postgres pg_dump -U aals_user aals > backup.sql

# 設定ファイルバックアップ
tar -czf aals-config-backup.tar.gz config/ .env
```

### アップデート

```bash
# 最新版にアップデート
git pull origin main
./scripts/deploy.sh --backup
```

## 🚨 トラブルシューティング

### よくある問題

#### 1. データベース接続エラー
```bash
# データベース状態確認
docker-compose exec postgres pg_isready -U aals_user

# 接続テスト
docker-compose exec aals python -c "from aals.core.config import get_config; print('DB OK')"
```

#### 2. API認証エラー
```bash
# 環境変数確認
docker-compose exec aals env | grep AALS_

# トークン有効性確認
docker-compose exec aals python -c "
from aals.integrations.slack_client import SlackAPIClient
client = SlackAPIClient()
print(client.verify_connection())
"
```

#### 3. メモリ不足
```bash
# メモリ使用量確認
docker-compose exec aals free -h

# サービス再起動
docker-compose restart aals
```

### ログレベル変更

```bash
# 詳細ログに変更
echo "AALS_LOG_LEVEL=DEBUG" >> .env
docker-compose restart aals
```

### パフォーマンス最適化

```bash
# ワーカー数調整
echo "AALS_WORKERS=8" >> .env

# 接続プール調整
echo "AALS_MAX_CONNECTIONS=200" >> .env

docker-compose restart aals
```

## 📞 サポート

- **Issues**: https://github.com/your-org/aals/issues
- **Documentation**: https://docs.your-org.com/aals
- **Support Email**: aals-support@your-org.com

## 📝 ライセンス

MIT License - 詳細は LICENSE ファイルを参照
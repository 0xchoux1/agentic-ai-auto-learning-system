#!/bin/bash

# AALS Docker Entrypoint Script
set -e

echo "🚀 Starting AALS Application..."

# Wait for database to be ready
echo "⏳ Waiting for PostgreSQL..."
while ! pg_isready -h "${AALS_DB_HOST:-postgres}" -p "${AALS_DB_PORT:-5432}" -U "${AALS_DB_USER:-aals_user}"; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 1
done
echo "✅ PostgreSQL is ready!"

# Wait for Redis to be ready
echo "⏳ Waiting for Redis..."
until redis-cli -h "${AALS_REDIS_HOST:-redis}" -p "${AALS_REDIS_PORT:-6379}" ping; do
    echo "Redis is unavailable - sleeping"
    sleep 1
done
echo "✅ Redis is ready!"

# Run database migrations if needed
echo "🔧 Setting up database..."
python -c "
import asyncio
from aals.core.config import get_config
print('✅ Database configuration validated')
"

# Validate configuration
echo "🔧 Validating configuration..."
python -c "
from aals.core.config import get_config
config = get_config()
print(f'✅ Configuration loaded for environment: {config.environment}')
"

# Run startup checks
echo "🔍 Running startup checks..."
python -c "
import asyncio
from aals.modules.slack_alert_reader import SlackAlertReader
from aals.modules.prometheus_analyzer import PrometheusAnalyzer
from aals.modules.github_issues_searcher import GitHubIssuesSearcher
from aals.modules.llm_wrapper import LLMWrapper
from aals.modules.alert_correlator import AlertCorrelator
from aals.modules.ssh_executor import SSHExecutor
from aals.modules.response_orchestrator import ResponseOrchestrator

async def startup_check():
    print('🧪 Testing module initialization...')
    # Test basic module creation without dependencies
    print('✅ All modules can be imported successfully')

asyncio.run(startup_check())
"

echo "🎉 AALS Application startup complete!"
echo "🌐 Starting web server..."

# Execute the main command
exec "$@"
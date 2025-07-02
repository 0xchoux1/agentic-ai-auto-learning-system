#!/bin/bash

# AALS Production Deployment Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 AALS Production Deployment Script"
echo "============================================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root for security reasons."
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo "🔍 Checking dependencies..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Dependencies check passed"

# Check if .env file exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp "$PROJECT_DIR/.env.production" "$PROJECT_DIR/.env"
    echo "📝 Please edit .env file with your actual configuration values:"
    echo "   - AALS_SECRET_KEY"
    echo "   - AALS_DB_PASSWORD"
    echo "   - AALS_REDIS_PASSWORD"
    echo "   - AALS_SLACK_TOKEN"
    echo "   - AALS_GITHUB_TOKEN"
    echo "   - AALS_CLAUDE_API_KEY"
    echo ""
    echo "❌ Deployment stopped. Please configure .env file and run again."
    exit 1
fi

echo "✅ Environment configuration found"

# Load environment variables
set -a
source "$PROJECT_DIR/.env"
set +a

# Validate required environment variables
required_vars=(
    "AALS_SECRET_KEY"
    "AALS_DB_PASSWORD"
    "AALS_REDIS_PASSWORD"
)

echo "🔒 Validating security configuration..."
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ] || [ "${!var}" = "your-secure-password-here" ] || [ "${!var}" = "your-super-secret-key-here-change-this-in-production" ]; then
        echo "❌ Required environment variable $var is not set or uses default value"
        echo "   Please update your .env file with secure values"
        exit 1
    fi
done

echo "✅ Security configuration validated"

# Create required directories
echo "📁 Creating required directories..."
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data/ssh_keys"
mkdir -p "$PROJECT_DIR/monitoring/grafana/provisioning"

# Set proper permissions
chmod 700 "$PROJECT_DIR/data/ssh_keys"

echo "✅ Directories created"

# Function to backup existing deployment
backup_deployment() {
    if [ -d "$PROJECT_DIR/backups" ]; then
        echo "💾 Creating backup of existing deployment..."
        BACKUP_DIR="$PROJECT_DIR/backups/backup-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        # Backup configuration and data
        cp -r "$PROJECT_DIR/config" "$BACKUP_DIR/" 2>/dev/null || true
        cp -r "$PROJECT_DIR/logs" "$BACKUP_DIR/" 2>/dev/null || true
        cp "$PROJECT_DIR/.env" "$BACKUP_DIR/" 2>/dev/null || true
        
        echo "✅ Backup created at $BACKUP_DIR"
    fi
}

# Function to deploy application
deploy_application() {
    echo "🚀 Starting AALS deployment..."
    
    cd "$PROJECT_DIR"
    
    # Pull latest images
    echo "📥 Pulling latest images..."
    docker-compose pull
    
    # Build application image
    echo "🔨 Building AALS application..."
    docker-compose build --no-cache aals
    
    # Start services
    echo "🎬 Starting services..."
    docker-compose up -d
    
    echo "⏳ Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    echo "🔍 Checking service health..."
    
    # Check database
    if docker-compose exec -T postgres pg_isready -U aals_user -d aals; then
        echo "✅ Database is healthy"
    else
        echo "❌ Database health check failed"
        return 1
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping > /dev/null; then
        echo "✅ Redis is healthy"
    else
        echo "❌ Redis health check failed"
        return 1
    fi
    
    # Check AALS application
    sleep 5
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ AALS application is healthy"
    else
        echo "⚠️  AALS application health check failed, but deployment may still be starting..."
        echo "   Check logs with: docker-compose logs aals"
    fi
    
    echo "🎉 Deployment completed successfully!"
}

# Function to show post-deployment information
show_deployment_info() {
    echo ""
    echo "🌐 AALS Application Information"
    echo "=================================="
    echo "Application URL: http://localhost:8000"
    echo "Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "Prometheus Metrics: http://localhost:9090"
    echo ""
    echo "📋 Useful commands:"
    echo "  View logs:           docker-compose logs -f aals"
    echo "  Stop services:       docker-compose down"
    echo "  Restart services:    docker-compose restart"
    echo "  Update application:  ./scripts/deploy.sh"
    echo ""
    echo "🔍 Monitoring:"
    echo "  Application health:  curl http://localhost:8000/health"
    echo "  Service status:      docker-compose ps"
    echo ""
}

# Main deployment flow
main() {
    echo "Starting deployment process..."
    
    # Optional: Create backup
    if [ "$1" = "--backup" ]; then
        backup_deployment
    fi
    
    # Deploy application
    deploy_application
    
    # Show information
    show_deployment_info
    
    echo "✅ AALS deployment completed successfully!"
}

# Run main function
main "$@"
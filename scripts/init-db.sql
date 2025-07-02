-- AALS Database Initialization Script

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create core tables for AALS system
CREATE TABLE IF NOT EXISTS aals_workflows (
    id SERIAL PRIMARY KEY,
    workflow_id VARCHAR(255) UNIQUE NOT NULL,
    incident_event_id VARCHAR(255),
    automation_level VARCHAR(50),
    status VARCHAR(50),
    current_stage VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_by VARCHAR(255),
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS aals_workflow_steps (
    id SERIAL PRIMARY KEY,
    step_id VARCHAR(255) UNIQUE NOT NULL,
    workflow_id VARCHAR(255) REFERENCES aals_workflows(workflow_id),
    stage VARCHAR(50),
    name VARCHAR(255),
    description TEXT,
    module VARCHAR(100),
    action VARCHAR(100),
    parameters JSONB DEFAULT '{}',
    status VARCHAR(50),
    result JSONB,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time FLOAT DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS aals_alert_contexts (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100),
    timestamp TIMESTAMP,
    severity VARCHAR(50),
    content JSONB,
    confidence FLOAT,
    correlation_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS aals_ssh_requests (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    command TEXT NOT NULL,
    target_host VARCHAR(255),
    target_user VARCHAR(255),
    permission_level VARCHAR(50),
    approval_status VARCHAR(50),
    requester VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    approved_at TIMESTAMP,
    executed_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS aals_audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255),
    action VARCHAR(100),
    resource VARCHAR(255),
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_workflows_status ON aals_workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON aals_workflows(created_at);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_id ON aals_workflow_steps(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_status ON aals_workflow_steps(status);
CREATE INDEX IF NOT EXISTS idx_alert_contexts_timestamp ON aals_alert_contexts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alert_contexts_source ON aals_alert_contexts(source);
CREATE INDEX IF NOT EXISTS idx_ssh_requests_status ON aals_ssh_requests(approval_status);
CREATE INDEX IF NOT EXISTS idx_ssh_requests_created_at ON aals_ssh_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON aals_audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON aals_audit_log(user_id);

-- Grant permissions to AALS user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aals_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aals_user;

-- Insert initial data or configuration if needed
INSERT INTO aals_workflows (workflow_id, status, current_stage, created_by) 
VALUES ('system-init', 'completed', 'initialization', 'system')
ON CONFLICT (workflow_id) DO NOTHING;

-- Create materialized view for workflow statistics (optional)
CREATE MATERIALIZED VIEW IF NOT EXISTS aals_workflow_stats AS
SELECT 
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds
FROM aals_workflows 
WHERE started_at IS NOT NULL 
GROUP BY status;

-- Create function to refresh workflow stats
CREATE OR REPLACE FUNCTION refresh_workflow_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW aals_workflow_stats;
END;
$$ LANGUAGE plpgsql;

COMMIT;
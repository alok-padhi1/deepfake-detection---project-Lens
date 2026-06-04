CREATE TABLE IF NOT EXISTS cases (
    case_id UUID PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    confidence_score FLOAT,
    decision_band VARCHAR(50),
    decision_label VARCHAR(50),
    media_type VARCHAR(50),
    modules_complete JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_value FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_baselines (
    module_name VARCHAR(100),
    media_type VARCHAR(50),
    baseline_mean FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (module_name, media_type)
);

CREATE TABLE IF NOT EXISTS model_versions (
    module_name VARCHAR(100),
    media_type VARCHAR(50),
    version VARCHAR(50),
    auc FLOAT,
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (module_name, media_type)
);

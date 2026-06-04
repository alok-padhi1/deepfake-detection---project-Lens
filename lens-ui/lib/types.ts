// lib/types.ts
export interface Case {
  case_id: string;
  status: "INGESTED" | "PROCESSING" | "COMPLETE" | "FAILED" | "SKIPPED";
  confidence_score: number;
  decision_band: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  decision_label: "AUTHENTIC" | "SUSPICIOUS" | "FAKE";
  media_type: "image" | "video" | "audio";
  modules_complete: string[];
  created_at: string;
  completed_at: string | null;
  filename?: string;
}

export interface MetricSummary {
  total_cases: number;
  synthetic_count: number;
  authentic_count: number;
  alert_rate_by_band: Record<string, number>;
  avg_latency: Record<string, number>;
  active_pipeline_count: number;
}

export interface HealthStatus {
  status: string;
  model_versions: Record<string, string>;
  kafka_lag: number;
  gpu_pool_utilization: number;
  uptime: number;
}

export interface User {
  sub: string;
  username: string;
  email: string;
  roles: string[];
  status: string;
  last_login?: string;
}

export interface DeployedModel {
  name: string;
  version: string;
  auc: number;
  deployed_at: string;
  status: "ACTIVE" | "DEPRECATED" | "RETRAINING";
  p50_latency_ms: number;
  p99_latency_ms: number;
}

export interface AuditEntry {
  entry_id: string;
  case_id: string;
  event_type: string;
  actor: string;
  timestamp: string;
  prev_hash: string;
  hash: string;
}

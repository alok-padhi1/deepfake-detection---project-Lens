# lens-infra/ — Complete Directory Tree
# LENS AI Forensic Detection Platform
# Parts 1–6 combined repo scaffold

lens-infra/
├── helmfile.yaml                          # Helmfile: all releases in dependency order
├── Makefile                               # Operator targets: init/validate/plan/apply/destroy
│
├── environments/
│   ├── staging/
│   │   ├── values.yaml                    # Staging env overrides
│   │   └── secrets.yaml                   # SOPS/AGE encrypted secrets (staging)
│   └── production/
│       ├── values.yaml                    # Production env overrides
│       └── secrets.yaml                   # SOPS/AGE encrypted secrets (production)
│
├── scripts/
│   └── rotate-secrets.sh                  # Credential rotation: MinIO/PG/Redis/Kafka
│
├── .github/
│   └── workflows/
│       └── deploy.yml                     # CI/CD: lint→validate→diff→staging→prod
│
├── docs/
│   ├── runbook.md                         # 12-step deploy + rollback + secret rotation
│   └── architecture.md                    # Namespace topology, data flow, mTLS map
│
# ── Helm Chart Values (per component) ─────────────────────────────────────────
├── helm/
│   ├── cert-manager/
│   │   ├── values.yaml
│   │   ├── values.staging.yaml
│   │   └── values.production.yaml
│   ├── calico/
│   │   ├── values.yaml
│   │   ├── values.staging.yaml
│   │   └── values.production.yaml
│   ├── gpu-operator/
│   │   ├── values.yaml
│   │   ├── values.staging.yaml
│   │   └── values.production.yaml
│   ├── keda/
│   │   ├── values.yaml
│   │   ├── values.staging.yaml
│   │   └── values.production.yaml
│   ├── istio/
│   │   ├── base-values.yaml
│   │   ├── istiod-values.yaml
│   │   ├── istiod-values.staging.yaml
│   │   ├── istiod-values.production.yaml
│   │   ├── gateway-values.yaml
│   │   ├── gateway-values.staging.yaml
│   │   └── gateway-values.production.yaml
│   ├── strimzi/
│   │   ├── values.yaml
│   │   ├── values.staging.yaml
│   │   └── values.production.yaml
│   ├── kube-prometheus-stack/
│   │   ├── values.yaml
│   │   ├── values.staging.yaml
│   │   └── values.production.yaml
│   ├── loki-stack/
│   │   ├── values.yaml
│   │   ├── values.staging.yaml
│   │   └── values.production.yaml
│   └── falco/
│       ├── values.yaml
│       ├── values.staging.yaml
│       └── values.production.yaml
│
# ── Kubernetes Manifests ───────────────────────────────────────────────────────
├── manifests/
│   │
│   ├── namespaces/                        # [Part 1]
│   │   ├── lens-api.yaml
│   │   ├── lens-gpu.yaml
│   │   ├── lens-kafka.yaml
│   │   ├── lens-storage.yaml
│   │   ├── lens-alert.yaml
│   │   └── lens-monitoring.yaml
│   │
│   ├── rbac/                              # [Part 1]
│   │   ├── cluster-roles.yaml
│   │   ├── cluster-role-bindings.yaml
│   │   ├── lens-api-sa.yaml
│   │   ├── lens-gpu-sa.yaml
│   │   ├── lens-kafka-sa.yaml
│   │   └── lens-storage-sa.yaml
│   │
│   ├── network-policies/                  # [Part 1]
│   │   ├── deny-all-default.yaml
│   │   ├── allow-lens-api.yaml
│   │   ├── allow-lens-gpu.yaml
│   │   ├── allow-lens-kafka.yaml
│   │   ├── allow-lens-storage.yaml
│   │   ├── allow-lens-alert.yaml
│   │   └── allow-monitoring-scrape.yaml
│   │
│   ├── pki/                               # [Part 1]
│   │   ├── cluster-issuer.yaml            # cert-manager ClusterIssuer (ACME/LetsEncrypt)
│   │   ├── lens-api-certificate.yaml
│   │   └── lens-kafka-certificate.yaml
│   │
│   ├── mesh/                              # [Part 2 — Istio]
│   │   ├── peer-authentication.yaml       # STRICT mTLS for all LENS namespaces
│   │   ├── destination-rules.yaml
│   │   ├── gateway.yaml                   # Istio Gateway (443 + 80→443 redirect)
│   │   ├── virtual-service-api.yaml
│   │   ├── virtual-service-ingest.yaml
│   │   └── authorization-policy.yaml
│   │
│   ├── kafka/                             # [Part 3 — Strimzi]
│   │   ├── kafka-node-pool.yaml
│   │   ├── kafka-cluster.yaml             # KafkaNodePool + Kafka CR (KRaft mode)
│   │   ├── kafka-topics.yaml              # media.raw, media.frames, results.detections, alerts.events
│   │   ├── kafka-users.yaml               # SCRAM-SHA-512 users per service
│   │   ├── schema-registry-deployment.yaml
│   │   ├── schema-registry-service.yaml
│   │   └── schema-registry-configmap.yaml
│   │
│   ├── storage/                           # [Part 4 — Storage]
│   │   ├── minio/
│   │   │   ├── namespace.yaml
│   │   │   ├── secret.yaml                # rootUser + rootPassword (SOPS-managed)
│   │   │   ├── statefulset.yaml
│   │   │   ├── service.yaml
│   │   │   ├── pvc.yaml
│   │   │   ├── lifecycle-policy.json
│   │   │   └── ingress.yaml
│   │   ├── postgres/
│   │   │   ├── patroni-configmap.yaml
│   │   │   ├── secret.yaml                # superuser + replication passwords
│   │   │   ├── statefulset.yaml
│   │   │   ├── service.yaml
│   │   │   ├── pvc.yaml
│   │   │   └── podisruptionbudget.yaml
│   │   ├── redis/
│   │   │   ├── secret.yaml
│   │   │   ├── statefulset.yaml
│   │   │   ├── service.yaml
│   │   │   └── sentinel-configmap.yaml
│   │   └── qdrant/
│   │       ├── secret.yaml
│   │       ├── statefulset.yaml
│   │       ├── service.yaml
│   │       └── pvc.yaml
│   │
│   ├── monitoring/                        # [Part 5 — Observability]
│   │   ├── prometheus-rules.yaml          # LENS custom alerting rules
│   │   ├── service-monitors.yaml          # PodMonitor / ServiceMonitor CRDs
│   │   ├── grafana-dashboards-cm.yaml     # Grafana dashboard ConfigMaps
│   │   ├── alertmanager-config.yaml       # Alertmanager routing (Slack/PD/email)
│   │   └── loki-datasource.yaml
│   │
│   ├── api/                               # [Part 5 — Application]
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── hpa.yaml
│   │   └── configmap.yaml
│   │
│   ├── ingest/                            # [Part 5 — Application]
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   │
│   ├── worker/                            # [Part 5 — GPU Workers]
│   │   ├── deployment.yaml                # GPU resource limits, tolerations
│   │   ├── scaledobject.yaml              # KEDA ScaledObject → Kafka lag
│   │   └── configmap.yaml
│   │
│   └── alert/                             # [Part 5 — Alert Engine]
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── scaledobject.yaml
│       └── configmap.yaml
│
# ── Avro Schemas ───────────────────────────────────────────────────────────────
└── schemas/
    └── avro/                              # [Part 3 — Kafka schemas]
        ├── MediaIngestEvent.avsc
        ├── MediaFrame.avsc
        ├── DetectionResult.avsc
        ├── FaceEmbedding.avsc
        └── AlertEvent.avsc

# docs/architecture.md
# LENS AI Forensic Detection Platform — Architecture Reference

## 1. Namespace Topology

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  Kubernetes Cluster (lens-cluster)                                               ║
║                                                                                  ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐  ║
║  │  cert-manager   │  │  calico-system  │  │         gpu-operator            │  ║
║  │  (ClusterIssuer │  │  (Tigera CNI +  │  │  (NVIDIA driver lifecycle,      │  ║
║  │   ACME/LetsEnc) │  │   NetworkPolicy)│  │   device plugin, DCGM exporter) │  ║
║  └─────────────────┘  └─────────────────┘  └─────────────────────────────────┘  ║
║                                                                                  ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐  ║
║  │  istio-system   │  │  istio-ingress  │  │            keda                 │  ║
║  │  (istiod,       │  │  (Gateway LB,   │  │  (ScaledObject operator,        │  ║
║  │   mTLS STRICT)  │  │   TLS offload)  │  │   metrics-server, webhooks)     │  ║
║  └─────────────────┘  └─────────────────┘  └─────────────────────────────────┘  ║
║                                                                                  ║
║  ┌───────────────────────────────────────────────────────────────────────────┐   ║
║  │  lens-kafka   [istio-injection=enabled]                                   │   ║
║  │                                                                           │   ║
║  │   ┌──────────────────────┐   ┌──────────────────────┐                    │   ║
║  │   │  Strimzi Kafka (×3)  │   │  Schema Registry (×2)│                    │   ║
║  │   │  SCRAM-SHA-512 auth  │   │  Avro / Confluent    │                    │   ║
║  │   │  TLS inter-broker    │   │  REST API            │                    │   ║
║  │   └──────────────────────┘   └──────────────────────┘                    │   ║
║  │   Topics: media.raw  media.frames  results.detections  alerts.events     │   ║
║  └───────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                  ║
║  ┌───────────────────────────────────────────────────────────────────────────┐   ║
║  │  lens-storage   [istio-injection=enabled]                                 │   ║
║  │                                                                           │   ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────────┐  │   ║
║  │  │ MinIO (×4)   │  │ Patroni PG   │  │ Redis    │  │ Qdrant (×3)      │  │   ║
║  │  │ distributed  │  │ HA (×3)      │  │ Sentinel │  │ vector store     │  │   ║
║  │  │ S3-compat    │  │ WAL archival │  │ (×3)     │  │ embeddings index │  │   ║
║  │  └──────────────┘  └──────────────┘  └──────────┘  └──────────────────┘  │   ║
║  └───────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                  ║
║  ┌─────────────────────────┐   ┌───────────────────────────────────────────┐    ║
║  │  lens-api               │   │  lens-gpu                                 │    ║
║  │  [istio-injection=ena.] │   │  [istio-injection=enabled]                │    ║
║  │                         │   │  [nodeSelector: gpu=present]              │    ║
║  │  ┌─────────────────┐    │   │                                           │    ║
║  │  │ API Gateway     │    │   │  ┌────────────────────────────────────┐   │    ║
║  │  │ (FastAPI + gRPC)│    │   │  │ GPU Worker (CUDA 12, TensorRT)     │   │    ║
║  │  └─────────────────┘    │   │  │ Deepfake Detection  │ Face Match   │   │    ║
║  │  ┌─────────────────┐    │   │  │ Object Detection    │ OCR Engine   │   │    ║
║  │  │ Ingest Service  │    │   │  └────────────────────────────────────┘   │    ║
║  │  │ (media upload,  │    │   │  KEDA ScaledObject → Kafka lag trigger     │    ║
║  │  │  pre-validation)│    │   └───────────────────────────────────────────┘    ║
║  │  └─────────────────┘    │                                                    ║
║  └─────────────────────────┘   ┌───────────────────────────────────────────┐    ║
║                                │  lens-alert                               │    ║
║  ┌─────────────────────────┐   │  [istio-injection=enabled]                │    ║
║  │  monitoring             │   │                                           │    ║
║  │                         │   │  ┌───────────────────────────────────┐    │    ║
║  │  Prometheus + Grafana   │   │  │ Alert Engine (webhook, email,     │    │    ║
║  │  Loki + Promtail        │   │  │ Slack, PagerDuty integrations)    │    │    ║
║  │  Alertmanager           │   │  └───────────────────────────────────┘    │    ║
║  └─────────────────────────┘   └───────────────────────────────────────────┘    ║
║                                                                                  ║
║  ┌────────────────┐  ┌────────────────┐  ┌───────────────────────────────────┐  ║
║  │  falco         │  │  strimzi-system│  │  kube-system                      │  ║
║  │  (eBPF runtime │  │  (Kafka        │  │  (CoreDNS, kube-proxy,            │  ║
║  │   security)    │  │   operator)    │  │   metrics-server)                 │  ║
║  └────────────────┘  └────────────────┘  └───────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Data Flow: Media Ingest → Analysis → Alert

```
  External Client
  (Browser / Mobile / RTSP Camera)
        │
        │  HTTPS (TLS 1.3)
        ▼
  ┌─────────────────────────────────┐
  │  istio-ingress (Gateway LB)     │
  │  TLS termination                │
  │  cert-manager managed cert      │
  └─────────────┬───────────────────┘
                │  mTLS (STRICT)
                ▼
  ┌─────────────────────────────────┐
  │  lens-api / Ingest Service      │
  │  • Validates media (type/size)  │
  │  • Assigns job UUID             │
  │  • Stores raw media → MinIO     │──────────────────┐
  │  • Publishes to Kafka           │                  │
  └─────────────┬───────────────────┘                  │
                │  Produce: media.raw                  │  S3 PUT
                │  (SCRAM-SHA-512 + TLS)               ▼
                ▼                          ┌───────────────────┐
  ┌────────────────────────────────┐       │  MinIO (lens-     │
  │  Kafka: media.raw topic        │       │  storage)         │
  │  3 partitions / RF=3           │       │  Bucket: media-   │
  │  Schema: MediaIngestEvent.avsc │       │  raw, media-proc  │
  └─────────────┬──────────────────┘       └───────────────────┘
                │  Consume (KEDA-scaled)
                ▼
  ┌─────────────────────────────────┐
  │  lens-gpu / GPU Worker          │
  │  • Downloads media from MinIO   │◄────────────────────────
  │  • Frame extraction (FFMPEG)    │
  │  • Deepfake detection (TRT)     │
  │  • Face recognition (ArcFace)   │
  │  • Object detection (YOLOv8)    │
  │  • Produces results to Kafka    │
  └─────────────┬───────────────────┘
                │  Produce: results.detections
                ▼
  ┌─────────────────────────────────┐
  │  Kafka: results.detections      │
  │  Schema: DetectionResult.avsc   │
  └──────┬──────────────────────────┘
         │
         ├─────────────────────────────────────────────┐
         │  Consume (lens-api)                         │  Consume (lens-alert)
         ▼                                             ▼
  ┌────────────────────────┐              ┌────────────────────────────┐
  │  PostgreSQL (Patroni)  │              │  Alert Engine              │
  │  • Store result rows   │              │  • Threshold evaluation    │
  │  • Job status update   │              │  • Webhook / Slack / PD    │
  └────────────────────────┘              │  • Produce: alerts.events  │
         │                                └────────────────────────────┘
         │  Write embeddings                           │
         ▼                                             ▼
  ┌────────────────────────┐              ┌────────────────────────────┐
  │  Qdrant (lens-storage) │              │  Kafka: alerts.events      │
  │  • Face embeddings     │              │  (audit trail, downstream  │
  │  • Similarity search   │              │   SIEM integration)        │
  └────────────────────────┘              └────────────────────────────┘
         │
         │  Cache hot results
         ▼
  ┌────────────────────────┐
  │  Redis Sentinel        │
  │  • Job status cache    │
  │  • Rate-limit counters │
  │  • Session tokens      │
  └────────────────────────┘
```

---

## 3. mTLS Trust Boundaries

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  LENS Istio mTLS Trust Domain: lens-mesh                                 │
  │                                                                          │
  │  Trust Anchor: Istio CA (istiod) — intermediate from cert-manager       │
  │  Certificate Rotation: automatic every 24h (SPIFFE x509 SVIDs)          │
  │                                                                          │
  │  ┌────────────────────────────────────────────────────────────────────┐  │
  │  │  STRICT mTLS Zone — all namespaces with istio-injection=enabled    │  │
  │  │                                                                    │  │
  │  │  lens-api  ←──mTLS──►  lens-kafka   ←──mTLS──►  lens-gpu          │  │
  │  │     │                      │                        │              │  │
  │  │     └──────mTLS────────────►  lens-storage  ◄───mTLS─┘            │  │
  │  │                             (MinIO, PG,                           │  │
  │  │                              Redis, Qdrant)                       │  │
  │  │                                                                    │  │
  │  │  lens-alert  ←──mTLS──►  lens-kafka                               │  │
  │  │  lens-alert  ←──mTLS──►  lens-storage                             │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  │                                                                          │
  │  ┌────────────────────────────────────────────────────────────────────┐  │
  │  │  PERMISSIVE / EXCLUDED — platform namespaces                       │  │
  │  │  (cert-manager, calico-system, gpu-operator, kube-system)          │  │
  │  │  Reason: bootstrap ordering; no Istio sidecar injected             │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  │                                                                          │
  │  ┌────────────────────────────────────────────────────────────────────┐  │
  │  │  Monitoring namespace (STRICT mTLS)                                │  │
  │  │  Prometheus scrape → all workloads via mTLS + PodMonitor CRDs     │  │
  │  │  Falco → Alertmanager via mTLS                                     │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  │                                                                          │
  │  External Boundary:                                                      │
  │  ┌─────────────────────────────────────────────────────────────────┐     │
  │  │  istio-ingress (Gateway)                                        │     │
  │  │  ┌──────────────────┐    ┌─────────────────────────────────┐   │     │
  │  │  │ TLS from client  │    │ mTLS to upstream services       │   │     │
  │  │  │ (ACME cert,      │───►│ (SPIFFE SVID, Istio-issued)     │   │     │
  │  │  │  LetsEncrypt)    │    └─────────────────────────────────┘   │     │
  │  │  └──────────────────┘                                          │     │
  │  └─────────────────────────────────────────────────────────────────┘     │
  │                                                                          │
  │  Kafka Internal TLS (separate from Istio mTLS):                          │
  │  ┌────────────────────────────────────────────────────────────────────┐  │
  │  │  Strimzi manages its own CA for inter-broker TLS + client TLS     │  │
  │  │  Listeners: PLAIN (internal), TLS+SCRAM (external within cluster) │  │
  │  │  Strimzi CA cert rotated every 365d (configurable)                │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. NetworkPolicy Isolation Model

```
  Namespace: lens-gpu
  ┌──────────────────────────────────────────────────────────┐
  │  Ingress: ALLOW from lens-kafka (consumer pull via mesh) │
  │  Egress:  ALLOW to lens-kafka   (produce results)        │
  │  Egress:  ALLOW to lens-storage (MinIO S3 GET/PUT)       │
  │  Egress:  ALLOW to monitoring   (Prometheus metrics)     │
  │  ALL OTHER ingress/egress: DENY                          │
  └──────────────────────────────────────────────────────────┘

  Namespace: lens-storage
  ┌──────────────────────────────────────────────────────────┐
  │  Ingress: ALLOW from lens-api, lens-gpu, lens-alert      │
  │  Ingress: ALLOW from monitoring (Prometheus scrape)      │
  │  Egress:  ALLOW DNS (kube-dns :53)                       │
  │  Egress:  ALLOW intra-namespace (PG replication, etc.)   │
  │  ALL OTHER ingress/egress: DENY                          │
  └──────────────────────────────────────────────────────────┘

  Namespace: lens-kafka
  ┌──────────────────────────────────────────────────────────┐
  │  Ingress: ALLOW from lens-api, lens-gpu, lens-alert      │
  │  Ingress: ALLOW from strimzi-system (operator)           │
  │  Ingress: ALLOW from monitoring (Prometheus scrape)      │
  │  Egress:  ALLOW intra-namespace (broker-to-broker)       │
  │  Egress:  ALLOW DNS                                      │
  │  ALL OTHER ingress/egress: DENY                          │
  └──────────────────────────────────────────────────────────┘
```

---

## 5. KEDA Autoscaling Triggers

| ScaledObject | Namespace | Trigger | Min | Max |
|---|---|---|---|---|
| `lens-gpu-worker` | `lens-gpu` | Kafka consumer lag: `media.raw` ≥ 100 msgs | 1 | 8 |
| `lens-ingest` | `lens-api` | Kafka consumer lag: `media.raw` ≥ 200 msgs | 1 | 4 |
| `lens-alert` | `lens-alert` | Kafka consumer lag: `results.detections` ≥ 50 msgs | 1 | 4 |
| `lens-api-gateway` | `lens-api` | Prometheus: `http_requests_per_second` ≥ 500 | 2 | 10 |

---

## 6. Component Version Matrix

| Component | Version | Chart Source |
|---|---|---|
| Kubernetes | ≥ 1.28 | Cloud provider / kubeadm |
| cert-manager | v1.14.5 | jetstack |
| Calico / Tigera | v3.27.3 | projectcalico |
| NVIDIA GPU Operator | v23.9.2 | nvidia |
| KEDA | 2.14.0 | kedacore |
| Istio | 1.21.2 | istio-release |
| Strimzi Kafka Operator | 0.40.0 | strimzi |
| Kafka (Strimzi-managed) | 3.7.0 | via Strimzi CR |
| kube-prometheus-stack | 58.2.2 | prometheus-community |
| Loki Stack | 2.10.2 | grafana |
| Falco | 3.8.7 | falcosecurity |
| MinIO | RELEASE.2024-04-06 | manifests (direct) |
| Patroni / PostgreSQL | 3.2.2 / 16.2 | manifests (direct) |
| Redis | 7.2 | manifests (direct) |
| Qdrant | 1.9.0 | manifests (direct) |

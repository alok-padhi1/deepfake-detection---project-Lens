# docs/runbook.md
# LENS AI Forensic Detection Platform — Operations Runbook

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Pre-Deployment Checklist](#2-pre-deployment-checklist)
3. [Deployment Steps (Ordered)](#3-deployment-steps-ordered)
4. [Rollback Procedures](#4-rollback-procedures)
5. [Secret Rotation](#5-secret-rotation)
6. [Disaster Recovery](#6-disaster-recovery)
7. [Monitoring and Alerting](#7-monitoring-and-alerting)
8. [Common Troubleshooting](#8-common-troubleshooting)

---

## 1. Prerequisites

### Required Tools and Minimum Versions

| Tool | Min Version | Install |
|---|---|---|
| `kubectl` | ≥ 1.28 | https://kubernetes.io/docs/tasks/tools/ |
| `helm` | ≥ 3.14 | https://helm.sh/docs/intro/install/ |
| `helmfile` | ≥ 0.162 | https://helmfile.readthedocs.io/en/latest/#installation |
| `kustomize` | ≥ 5.3 | https://kubectl.docs.kubernetes.io/installation/kustomize/ |
| `strimzi-kafka-cli` | ≥ 0.3 | `pip install strimzi-kafka-cli` |
| `kubeval` | ≥ 0.16 | https://www.kubeval.com/installation/ |
| `sops` | ≥ 3.8 | https://github.com/getsops/sops/releases |
| `age` | ≥ 1.1 | https://github.com/FiloSottile/age/releases |
| `jq` | ≥ 1.6 | https://jqlang.github.io/jq/download/ |
| `curl` | any | OS package manager |

### Verify All Prerequisites

```bash
make init
```

This command will check for all required tools and add Helm repositories.

### Required Helm Plugins

```bash
helm plugin install https://github.com/databus23/helm-diff
helm plugin install https://github.com/jkroepke/helm-secrets
```

### Cluster Requirements

- Kubernetes ≥ 1.28 with CRI-O or containerd runtime
- At least 3 worker nodes (8 vCPU / 32 GB RAM each minimum)
- At least 1 GPU node (NVIDIA T4 or better)
- Default StorageClass configured (e.g., `gp3`, `standard-ssd`)
- LoadBalancer support (cloud LB or MetalLB on bare-metal)
- OIDC issuer configured for service account token projection

---

## 2. Pre-Deployment Checklist

- [ ] Cluster kubeconfig is set: `export KUBECONFIG=/path/to/kubeconfig`
- [ ] Cluster is reachable: `kubectl cluster-info`
- [ ] Sufficient node capacity: `kubectl describe nodes`
- [ ] SOPS AGE key available: `export SOPS_AGE_KEY_FILE=~/.age-key.txt`
- [ ] Domain delegated and DNS resolvable
- [ ] Container image registry accessible from cluster nodes
- [ ] All required secrets populated in `environments/<env>/secrets.yaml` (SOPS-encrypted)
- [ ] StorageClass `gp3` (or equivalent) is the default
- [ ] GPU node labeled: `kubectl label node <node> nvidia.com/gpu=present`

---

## 3. Deployment Steps (Ordered)

> **Rule:** Wait for each step to be `Ready` before proceeding to the next.
> All commands default to staging. Append `ENV=production` for production.

---

### Step 1 — cert-manager

```bash
helmfile -e staging apply -l component=pki

# Verify
kubectl wait --for=condition=Ready pods \
  -n cert-manager -l app.kubernetes.io/instance=cert-manager \
  --timeout=120s

kubectl get clusterissuer -A   # Should be empty until step 1b
```

**Step 1b — Apply ClusterIssuer (after cert-manager is Ready)**

```bash
kubectl apply -f manifests/pki/cluster-issuer.yaml

# Verify ClusterIssuer is ready
kubectl describe clusterissuer letsencrypt-staging
```

---

### Step 2 — Calico CNI

```bash
helmfile -e staging apply -l component=networking

# Verify tigera-operator is running
kubectl wait --for=condition=Ready pods \
  -n calico-system -l k8s-app=tigera-operator \
  --timeout=180s

# Apply LENS network policies
kubectl apply -f manifests/network-policies/

# Verify calico nodes are ready
kubectl get tigerastatus
```

---

### Step 3 — GPU Operator

```bash
helmfile -e staging apply -l component=gpu

# Verify GPU operator components
kubectl wait --for=condition=Ready pods \
  -n gpu-operator \
  --timeout=600s

# Verify GPU is discoverable on node
kubectl describe node <gpu-node> | grep nvidia.com/gpu
```

> **Note:** Driver installation can take 5–10 minutes on first deploy.

---

### Step 4 — KEDA (Event-Driven Autoscaler)

```bash
helmfile -e staging apply -l component=autoscaling

kubectl wait --for=condition=Ready pods \
  -n keda -l app=keda-operator \
  --timeout=120s

# Verify KEDA API is registered
kubectl api-resources | grep scaledobject
```

---

### Step 5 — Istio Service Mesh

Deploy in sub-steps: base → istiod → ingressgateway

```bash
# 5a — CRDs and base
helmfile -e staging apply -l component=istio

# Wait for istiod
kubectl wait --for=condition=Ready pods \
  -n istio-system -l app=istiod \
  --timeout=180s

# 5b — Apply LENS Gateways and VirtualServices
kubectl apply -f manifests/mesh/

# Verify PeerAuthentication (mTLS STRICT)
kubectl get peerauthentication -A

# Verify Gateway
kubectl get gateway -n istio-ingress
```

---

### Step 6 — Strimzi Kafka Operator

```bash
helmfile -e staging apply -l component=kafka

kubectl wait --for=condition=Ready pods \
  -n strimzi-system -l name=strimzi-cluster-operator \
  --timeout=180s

# 6b — Deploy Kafka cluster, Schema Registry, topics, users
kubectl apply -f manifests/kafka/

# Wait for Kafka cluster to be ready (can take 3-5 min)
kubectl wait kafka/lens-kafka -n lens-kafka \
  --for=condition=Ready --timeout=600s

# 6c — Seed Avro schemas
make seed-schemas

# Verify topics
skaf kafka-topics --list \
  --bootstrap-server lens-kafka-kafka-bootstrap.lens-kafka.svc.cluster.local:9093
```

---

### Step 7 — kube-prometheus-stack

```bash
export GRAFANA_ADMIN_PASSWORD="$(openssl rand -base64 32)"

helmfile -e staging apply -l component=prometheus

kubectl wait --for=condition=Ready pods \
  -n monitoring -l app.kubernetes.io/name=prometheus \
  --timeout=300s

# Apply LENS custom PrometheusRules and dashboards
kubectl apply -f manifests/monitoring/

# Access Grafana
make port-forward-grafana
# Open http://localhost:3000 → admin / $GRAFANA_ADMIN_PASSWORD
```

---

### Step 8 — Loki Stack

```bash
helmfile -e staging apply -l component=loki

kubectl wait --for=condition=Ready pods \
  -n monitoring -l app=loki \
  --timeout=180s

# Verify Promtail is running on all nodes
kubectl get pods -n monitoring -l app=promtail
```

---

### Step 9 — Falco Runtime Security

```bash
helmfile -e staging apply -l component=falco

kubectl wait --for=condition=Ready pods \
  -n falco -l app.kubernetes.io/name=falco \
  --timeout=300s

# Test Falco is emitting events
kubectl logs -n falco -l app.kubernetes.io/name=falco \
  --tail=20 --follow=false

# Verify Falcosidekick → Alertmanager connectivity
kubectl logs -n falco -l app.kubernetes.io/name=falcosidekick \
  --tail=20
```

---

### Step 10 — Namespaces and RBAC

```bash
kubectl apply -f manifests/namespaces/
kubectl apply -f manifests/rbac/

# Verify namespace labels (for Istio injection and PSA)
kubectl get ns lens-api lens-kafka lens-storage lens-gpu lens-monitoring \
  --show-labels
```

---

### Step 11 — Storage Layer (MinIO, PostgreSQL, Redis, Qdrant)

```bash
# Apply secrets first (must be done manually or via sealed-secrets)
kubectl apply -f manifests/storage/minio/secret.yaml
kubectl apply -f manifests/storage/postgres/secret.yaml
kubectl apply -f manifests/storage/redis/secret.yaml
kubectl apply -f manifests/storage/qdrant/secret.yaml

# Deploy storage workloads
kubectl apply -f manifests/storage/

# Verify
kubectl wait --for=condition=Ready pods \
  -n lens-storage --all --timeout=300s

# Port-forward to verify connectivity
make port-forward-postgres
make port-forward-redis
make port-forward-minio
```

---

### Step 12 — LENS Application Services (GPU Workers Last)

```bash
# API Gateway and ingest services
kubectl apply -f manifests/api/
kubectl apply -f manifests/ingest/
kubectl apply -f manifests/alert/

# GPU-accelerated analysis workers (last — GPU node must be ready)
kubectl apply -f manifests/worker/

# Verify all LENS pods
kubectl get pods -A -l app.kubernetes.io/part-of=lens

# Check KEDA ScaledObjects are active
kubectl get scaledobject -A
```

---

## 4. Rollback Procedures

### cert-manager Rollback

```bash
helm rollback cert-manager -n cert-manager [REVISION]
kubectl rollout status deploy/cert-manager -n cert-manager
```

### Calico Rollback

```bash
helm rollback calico -n calico-system [REVISION]
# If CNI is broken, drain and recycle nodes:
kubectl drain <node> --ignore-daemonsets --delete-emptydir-data
```

### GPU Operator Rollback

```bash
helm rollback gpu-operator -n gpu-operator [REVISION]
# Driver uninstall is automatic via operator
kubectl rollout status ds/nvidia-device-plugin-daemonset -n gpu-operator
```

### KEDA Rollback

```bash
helm rollback keda -n keda [REVISION]
# ScaledObjects remain until deleted; HPA is auto-recreated by KEDA on re-deploy
```

### Istio Rollback

```bash
# Rollback istiod first, then gateway, then base
helm rollback istiod      -n istio-system  [REVISION]
helm rollback istio-ingressgateway -n istio-ingress [REVISION]
helm rollback istio-base  -n istio-system  [REVISION]

# Force re-inject sidecars
kubectl rollout restart deployment -n lens-api
kubectl rollout restart deployment -n lens-kafka
```

### Strimzi Rollback

```bash
helm rollback strimzi-kafka-operator -n strimzi-system [REVISION]
# Kafka cluster itself is a CR — rollback via manifest revert:
git checkout HEAD~1 -- manifests/kafka/kafka-cluster.yaml
kubectl apply -f manifests/kafka/kafka-cluster.yaml
```

### kube-prometheus-stack Rollback

```bash
helm rollback kube-prometheus-stack -n monitoring [REVISION]
kubectl rollout status deploy/kube-prometheus-stack-grafana -n monitoring
```

### Loki Rollback

```bash
helm rollback loki-stack -n monitoring [REVISION]
```

### Falco Rollback

```bash
helm rollback falco -n falco [REVISION]
# If eBPF driver fails to load after rollback, switch to legacy driver:
helm upgrade falco falcosecurity/falco -n falco \
  --set driver.kind=module --reuse-values
```

### Application Service Rollback

```bash
# Rollback a specific deployment
kubectl rollout undo deployment/<name> -n <namespace>
kubectl rollout status deployment/<name> -n <namespace>

# List rollout history
kubectl rollout history deployment/<name> -n <namespace>
```

---

## 5. Secret Rotation

### MinIO Root Password

```bash
make rotate-secrets ENV=staging  # runs scripts/rotate-secrets.sh staging minio

# Manually: update SOPS secrets.yaml, then re-apply
helmfile -e staging apply -l component=minio
```

**Verify:** MinIO console accessible at `http://localhost:9001` with new credentials.

### PostgreSQL Superuser Password

```bash
make rotate-secrets ENV=staging  # with "postgres" subcommand

# Manually rotate via psql:
kubectl exec -n lens-storage patroni-postgres-0 -- \
  psql -U postgres -c "ALTER USER postgres PASSWORD 'NEW_PASS';"

# Update secret:
kubectl patch secret postgres-superuser -n lens-storage \
  --type='json' \
  -p="[{\"op\":\"replace\",\"path\":\"/data/password\",\"value\":\"$(echo -n 'NEW_PASS' | base64 -w0)\"}]"

# Restart Patroni to pick up new password (rolling):
kubectl rollout restart statefulset/patroni-postgres -n lens-storage
```

**Verify:** `kubectl exec -n lens-storage patroni-postgres-0 -- psql -U postgres -c '\l'`

### Redis Password

```bash
make rotate-secrets ENV=staging  # with "redis" subcommand

# Verify after rolling restart:
kubectl exec -n lens-storage redis-0 -- \
  redis-cli -a NEW_PASS ping
```

### Kafka SCRAM Credentials

```bash
make rotate-secrets ENV=staging  # with "kafka" subcommand

# Strimzi reconciles SCRAM-SHA-512 automatically when KafkaUser CR or Secret is updated.
# Verify:
kubectl get kafkauser -n lens-kafka
kubectl describe kafkauser lens-ingest -n lens-kafka
```

**Post-rotation:**
1. Update SOPS-encrypted `environments/<env>/secrets.yaml`
2. Commit and push — CI pipeline will re-apply
3. Verify all dependent services restarted and reconnected

---

## 6. Disaster Recovery

### Full Cluster Restore

1. Restore cluster from cloud-provider snapshot or etcd backup
2. Validate storage PVs are intact: `kubectl get pv`
3. Re-run deployment from Step 1 in sequence
4. Restore PostgreSQL from WAL-G backup:
   ```bash
   kubectl exec -n lens-storage patroni-postgres-0 -- \
     patronictl -c /etc/patroni/patroni.yaml restore lens-postgres \
     --master patroni-postgres-0 --force
   ```
5. Restore MinIO data from object store backup or S3 mirror

### etcd Backup

```bash
# Run from a control-plane node
ETCDCTL_API=3 etcdctl snapshot save /backup/etcd-$(date +%Y%m%d).db \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key
```

---

## 7. Monitoring and Alerting

| Component | URL (port-forward) | Credentials |
|---|---|---|
| Grafana | http://localhost:3000 | admin / `$GRAFANA_ADMIN_PASSWORD` |
| Prometheus | http://localhost:9090 | None |
| Alertmanager | http://localhost:9093 | None |
| MinIO Console | http://localhost:9001 | rootUser / rootPassword |
| Kafka UI | http://localhost:8080 | None (internal) |

### Key Dashboards

- **LENS Platform Overview** — GPU utilization, inference latency, alert pipeline throughput
- **Kafka Consumer Lag** — Lag per consumer group (lens-worker, lens-alert)
- **Istio Service Mesh** — mTLS coverage, request rates, error rates
- **Node GPU Metrics** — DCGM exporter metrics: SM utilization, VRAM, temp

---

## 8. Common Troubleshooting

### Pods stuck in `Init:0/1`

```bash
kubectl describe pod <pod> -n <ns>
# Check: ImagePullBackOff → registry credentials
# Check: Init container waiting for DNS → Calico not ready
```

### Istio sidecar not injecting

```bash
# Verify namespace has injection label
kubectl get ns <ns> --show-labels | grep istio-injection

# Add label if missing
kubectl label ns <ns> istio-injection=enabled
kubectl rollout restart deployment -n <ns>
```

### Kafka consumer lag growing

```bash
kubectl exec -n lens-kafka \
  lens-kafka-kafka-0 -- \
  bin/kafka-consumer-groups.sh \
    --bootstrap-server localhost:9092 \
    --describe --group lens-worker
```

Scale up GPU workers:
```bash
kubectl scale deployment lens-gpu-worker -n lens-gpu --replicas=4
```

### GPU worker OOMKilled

```bash
kubectl describe pod <worker-pod> -n lens-gpu | grep -A5 "Last State"
# Increase memory limit in manifests/worker/deployment.yaml
# Or reduce batch size via env var LENS_BATCH_SIZE
```

### cert-manager CertificateRequest stuck

```bash
kubectl describe certificaterequest -n <ns>
# Common: ACME DNS challenge timeout → verify DNS delegation
# Common: Rate limit hit → use letsencrypt-staging issuer first
```

### Schema Registry unable to connect to Kafka

```bash
kubectl logs -n lens-kafka -l app=schema-registry --tail=50
# Verify mTLS: schema-registry must have Istio sidecar and valid peer cert
kubectl get peerauthentication -n lens-kafka
```

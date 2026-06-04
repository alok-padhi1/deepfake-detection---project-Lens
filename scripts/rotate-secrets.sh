#!/usr/bin/env bash
# scripts/rotate-secrets.sh
# LENS Platform — Secret Rotation Script
# Usage: bash scripts/rotate-secrets.sh [staging|production]
# Rotates credentials for: MinIO, PostgreSQL, Redis, Kafka SCRAM

set -euo pipefail

ENV="${1:-staging}"
KUBECTL="kubectl"
OPENSSL="openssl"
DATE=$(date +%Y%m%d%H%M%S)

# ── Color helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { printf "${CYAN}[INFO]${RESET}  %s\n" "$*"; }
success() { printf "${GREEN}[OK]${RESET}    %s\n" "$*"; }
warn()    { printf "${YELLOW}[WARN]${RESET}  %s\n" "$*"; }
error()   { printf "${RED}[ERROR]${RESET} %s\n" "$*" >&2; exit 1; }

gen_password() { ${OPENSSL} rand -base64 32 | tr -d '=+/' | cut -c1-32; }

# ── MinIO Rotation ─────────────────────────────────────────────────────────────
rotate_minio() {
  info "Rotating MinIO credentials [${ENV}]..."
  local ns="lens-storage"
  local secret_name="minio-root-credentials"
  local new_pass; new_pass=$(gen_password)

  # Backup existing secret
  ${KUBECTL} get secret "${secret_name}" -n "${ns}" \
    -o yaml > "/tmp/backup-${secret_name}-${DATE}.yaml" 2>/dev/null || true

  # Update secret
  ${KUBECTL} patch secret "${secret_name}" -n "${ns}" \
    --type='json' \
    -p="[{\"op\":\"replace\",\"path\":\"/data/rootPassword\",\"value\":\"$(echo -n "${new_pass}" | base64 -w0)\"}]"

  # Trigger MinIO pod restart
  ${KUBECTL} rollout restart statefulset/minio -n "${ns}" || true

  success "MinIO password rotated. New password written to /tmp/minio-newpass-${DATE}.txt (delete after use)"
  echo "${new_pass}" > "/tmp/minio-newpass-${DATE}.txt"
  chmod 600 "/tmp/minio-newpass-${DATE}.txt"
}

# ── PostgreSQL Rotation ────────────────────────────────────────────────────────
rotate_postgres() {
  info "Rotating PostgreSQL superuser password [${ENV}]..."
  local ns="lens-storage"
  local secret_name="postgres-superuser"
  local new_pass; new_pass=$(gen_password)

  ${KUBECTL} get secret "${secret_name}" -n "${ns}" \
    -o yaml > "/tmp/backup-${secret_name}-${DATE}.yaml" 2>/dev/null || true

  # Update Kubernetes secret
  ${KUBECTL} patch secret "${secret_name}" -n "${ns}" \
    --type='json' \
    -p="[{\"op\":\"replace\",\"path\":\"/data/password\",\"value\":\"$(echo -n "${new_pass}" | base64 -w0)\"}]"

  # Apply inside PostgreSQL via patronictl
  local primary; primary=$(${KUBECTL} exec -n "${ns}" \
    "$(${KUBECTL} get pod -n "${ns}" -l role=master -o jsonpath='{.items[0].metadata.name}')" \
    -- patronictl -c /etc/patroni/patroni.yaml list --format tsv 2>/dev/null \
    | grep Leader | awk '{print $1}' || echo "patroni-postgres-0")

  ${KUBECTL} exec -n "${ns}" "${primary}" -- \
    psql -U postgres -c "ALTER USER postgres PASSWORD '${new_pass}';" || \
    warn "Could not update password in-cluster. Update manually via psql."

  success "PostgreSQL password rotated. Backup at /tmp/backup-${secret_name}-${DATE}.yaml"
}

# ── Redis Rotation ─────────────────────────────────────────────────────────────
rotate_redis() {
  info "Rotating Redis password [${ENV}]..."
  local ns="lens-storage"
  local secret_name="redis-secret"
  local new_pass; new_pass=$(gen_password)

  ${KUBECTL} get secret "${secret_name}" -n "${ns}" \
    -o yaml > "/tmp/backup-${secret_name}-${DATE}.yaml" 2>/dev/null || true

  ${KUBECTL} patch secret "${secret_name}" -n "${ns}" \
    --type='json' \
    -p="[{\"op\":\"replace\",\"path\":\"/data/redis-password\",\"value\":\"$(echo -n "${new_pass}" | base64 -w0)\"}]"

  ${KUBECTL} rollout restart statefulset/redis -n "${ns}" || true
  success "Redis password rotated."
}

# ── Kafka SCRAM Rotation ───────────────────────────────────────────────────────
rotate_kafka_scram() {
  info "Rotating Kafka SCRAM credentials [${ENV}]..."
  local ns="lens-kafka"
  local new_pass; new_pass=$(gen_password)
  local users=("lens-ingest" "lens-worker" "lens-alert" "lens-api")

  for user in "${users[@]}"; do
    local secret_name="kafka-scram-${user}"
    ${KUBECTL} get secret "${secret_name}" -n "${ns}" \
      -o yaml > "/tmp/backup-${secret_name}-${DATE}.yaml" 2>/dev/null || true

    ${KUBECTL} patch secret "${secret_name}" -n "${ns}" \
      --type='json' \
      -p="[{\"op\":\"replace\",\"path\":\"/data/password\",\"value\":\"$(echo -n "${new_pass}" | base64 -w0)\"}]"

    # Update KafkaUser CR — Strimzi will reconcile SCRAM credentials automatically
    ${KUBECTL} annotate kafkauser "${user}" -n "${ns}" \
      "lens.io/secret-rotated=$(date -u +%FT%TZ)" --overwrite || true

    success "Kafka SCRAM password rotated for user: ${user}"
    echo "${user}:${new_pass}" >> "/tmp/kafka-scram-newpass-${DATE}.txt"
  done
  chmod 600 "/tmp/kafka-scram-newpass-${DATE}.txt"
  info "New Kafka SCRAM passwords at /tmp/kafka-scram-newpass-${DATE}.txt (delete after updating dependent services)"
}

# ── Main ───────────────────────────────────────────────────────────────────────
main() {
  printf "\n${BOLD}${CYAN}LENS Secret Rotation — Environment: ${ENV}${RESET}\n\n"

  case "${2:-all}" in
    minio)    rotate_minio ;;
    postgres) rotate_postgres ;;
    redis)    rotate_redis ;;
    kafka)    rotate_kafka_scram ;;
    all)
      rotate_minio
      rotate_postgres
      rotate_redis
      rotate_kafka_scram
      ;;
    *) error "Unknown component: ${2}. Use: minio|postgres|redis|kafka|all" ;;
  esac

  printf "\n${GREEN}${BOLD}Rotation complete.${RESET}\n"
  printf "${YELLOW}IMPORTANT: Update SOPS-encrypted secrets files and re-run helmfile apply.${RESET}\n\n"
}

main "$@"

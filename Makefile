# Makefile — LENS AI Forensic Detection Platform
# Targets: init, validate, plan, apply, destroy, port-forward, seed-schemas, rotate-secrets

SHELL            := /bin/bash
.DEFAULT_GOAL    := help
ENV              ?= staging
HELMFILE         := helmfile
HELM             := helm
KUBECTL          := kubectl
KUBEVAL          := kubeval
KUSTOMIZE        := kustomize
SCHEMA_REGISTRY  ?= http://localhost:8081
MANIFESTS_DIR    := manifests
HELM_DIR         := helm
SCHEMAS_DIR      := schemas/avro
SCRIPTS_DIR      := scripts

# ── ANSI color helpers ────────────────────────────────────────────────────────
BOLD    := \033[1m
RESET   := \033[0m
GREEN   := \033[32m
YELLOW  := \033[33m
RED     := \033[31m
CYAN    := \033[36m

.PHONY: help init validate plan apply destroy \
        port-forward port-forward-postgres port-forward-redis \
        port-forward-minio port-forward-grafana \
        seed-schemas rotate-secrets clean

# ─────────────────────────────────────────────────────────────────────────────
help: ## Show this help menu
	@printf "$(BOLD)$(CYAN)LENS Platform — Makefile Targets$(RESET)\n"
	@printf "$(YELLOW)Usage:$(RESET) make $(BOLD)<target>$(RESET) [ENV=staging|production]\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'

# ─────────────────────────────────────────────────────────────────────────────
init: ## Check all prerequisite tools and versions
	@printf "$(BOLD)Checking prerequisites...$(RESET)\n"
	@command -v kubectl >/dev/null 2>&1 || { echo "$(RED)ERROR: kubectl not found$(RESET)"; exit 1; }
	@command -v helm    >/dev/null 2>&1 || { echo "$(RED)ERROR: helm not found$(RESET)"; exit 1; }
	@command -v helmfile>/dev/null 2>&1 || { echo "$(RED)ERROR: helmfile not found$(RESET)"; exit 1; }
	@command -v kustomize>/dev/null 2>&1|| { echo "$(RED)ERROR: kustomize not found$(RESET)"; exit 1; }
	@command -v kubeval >/dev/null 2>&1 || { echo "$(YELLOW)WARN: kubeval not found (validation will be skipped)$(RESET)"; }
	@command -v jq      >/dev/null 2>&1 || { echo "$(YELLOW)WARN: jq not found (schema seeding may fail)$(RESET)"; }
	@command -v curl    >/dev/null 2>&1 || { echo "$(RED)ERROR: curl not found$(RESET)"; exit 1; }
	@printf "\n$(BOLD)Tool versions:$(RESET)\n"
	@kubectl version --client --short 2>/dev/null || kubectl version --client
	@helm version --short
	@$(HELMFILE) version
	@$(KUSTOMIZE) version
	@printf "\n$(GREEN)All prerequisites satisfied.$(RESET)\n"
	@printf "\n$(BOLD)Adding Helm repos...$(RESET)\n"
	@$(HELMFILE) repos
	@printf "$(GREEN)Repos synchronized.$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
validate: ## Helm lint all charts + kubeval all rendered manifests
	@printf "$(BOLD)$(CYAN)== Helm Lint ==$(RESET)\n"
	@for chart_dir in $(HELM_DIR)/*/; do \
		chart=$$(basename $$chart_dir); \
		printf "  Linting $(BOLD)$$chart$(RESET)... "; \
		$(HELM) lint $$chart_dir --quiet && printf "$(GREEN)OK$(RESET)\n" \
			|| { printf "$(RED)FAIL$(RESET)\n"; FAILED=1; }; \
	done; \
	if [ "$$FAILED" = "1" ]; then echo "$(RED)Lint failures detected$(RESET)"; exit 1; fi
	@printf "\n$(BOLD)$(CYAN)== Kubeval Manifests ==$(RESET)\n"
	@if command -v kubeval >/dev/null 2>&1; then \
		find $(MANIFESTS_DIR) -name "*.yaml" -not -name "*secret*" | \
			xargs $(KUBEVAL) \
				--kubernetes-version 1.28.0 \
				--strict \
				--ignore-missing-schemas \
				--schema-location https://raw.githubusercontent.com/yannh/kubernetes-json-schema/master/ \
		&& printf "$(GREEN)Kubeval passed.$(RESET)\n"; \
	else \
		printf "$(YELLOW)kubeval not installed — skipping manifest validation.$(RESET)\n"; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
plan: ## Diff helmfile releases against the current cluster state (ENV=staging|production)
	@printf "$(BOLD)$(CYAN)== Helmfile Diff [env=$(ENV)] ==$(RESET)\n"
	@$(HELMFILE) -e $(ENV) diff --suppress-secrets --context 5

# ─────────────────────────────────────────────────────────────────────────────
apply: ## Apply all Helm releases to the cluster (ENV=staging|production)
	@printf "$(BOLD)$(CYAN)== Helmfile Apply [env=$(ENV)] ==$(RESET)\n"
	@printf "$(YELLOW)Applying to environment: $(BOLD)$(ENV)$(RESET)\n"
	@$(HELMFILE) -e $(ENV) apply --suppress-secrets
	@printf "\n$(GREEN)Apply complete.$(RESET)\n"
	@$(MAKE) _post-apply-verify

_post-apply-verify:
	@printf "$(BOLD)Post-apply verification...$(RESET)\n"
	@$(KUBECTL) wait --for=condition=Ready pods \
		-n cert-manager -l app.kubernetes.io/instance=cert-manager \
		--timeout=120s
	@$(KUBECTL) wait --for=condition=Ready pods \
		-n istio-system -l app=istiod \
		--timeout=120s
	@$(KUBECTL) wait --for=condition=Ready pods \
		-n strimzi-system -l name=strimzi-cluster-operator \
		--timeout=120s
	@printf "$(GREEN)Core platform components healthy.$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
destroy: ## Destroy all Helm releases (ENV=staging|production) — DESTRUCTIVE
	@printf "$(RED)$(BOLD)WARNING: This will DESTROY all LENS releases in [$(ENV)].$(RESET)\n"
	@read -p "Type the environment name to confirm ($(ENV)): " confirm && \
		[ "$$confirm" = "$(ENV)" ] || { echo "Aborted."; exit 1; }
	@$(HELMFILE) -e $(ENV) destroy
	@printf "$(RED)Destroy complete.$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Port-forward targets (run in background, kill with: make kill-port-forwards)
# ─────────────────────────────────────────────────────────────────────────────
port-forward: port-forward-postgres port-forward-redis port-forward-minio port-forward-grafana ## Forward all key services

port-forward-postgres: ## Forward PostgreSQL → localhost:5432
	@printf "$(CYAN)Forwarding PostgreSQL → 127.0.0.1:5432$(RESET)\n"
	@$(KUBECTL) port-forward -n lens-storage \
		svc/patroni-postgres 5432:5432 \
		--address 127.0.0.1 &
	@echo $$! > /tmp/pf-postgres.pid
	@printf "$(GREEN)PID: $$(cat /tmp/pf-postgres.pid)$(RESET)\n"

port-forward-redis: ## Forward Redis → localhost:6379
	@printf "$(CYAN)Forwarding Redis → 127.0.0.1:6379$(RESET)\n"
	@$(KUBECTL) port-forward -n lens-storage \
		svc/redis-master 6379:6379 \
		--address 127.0.0.1 &
	@echo $$! > /tmp/pf-redis.pid
	@printf "$(GREEN)PID: $$(cat /tmp/pf-redis.pid)$(RESET)\n"

port-forward-minio: ## Forward MinIO API → localhost:9000, Console → localhost:9001
	@printf "$(CYAN)Forwarding MinIO API → 127.0.0.1:9000, Console → 127.0.0.1:9001$(RESET)\n"
	@$(KUBECTL) port-forward -n lens-storage \
		svc/minio 9000:9000 9001:9001 \
		--address 127.0.0.1 &
	@echo $$! > /tmp/pf-minio.pid
	@printf "$(GREEN)PID: $$(cat /tmp/pf-minio.pid)$(RESET)\n"

port-forward-grafana: ## Forward Grafana → localhost:3000
	@printf "$(CYAN)Forwarding Grafana → 127.0.0.1:3000$(RESET)\n"
	@$(KUBECTL) port-forward -n monitoring \
		svc/kube-prometheus-stack-grafana 3000:80 \
		--address 127.0.0.1 &
	@echo $$! > /tmp/pf-grafana.pid
	@printf "$(GREEN)PID: $$(cat /tmp/pf-grafana.pid)$(RESET)\n"

kill-port-forwards: ## Kill all background port-forward processes
	@for pidfile in /tmp/pf-*.pid; do \
		[ -f "$$pidfile" ] && kill $$(cat $$pidfile) 2>/dev/null && rm "$$pidfile"; \
	done
	@printf "$(GREEN)All port-forwards killed.$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
seed-schemas: ## Register all Avro schemas with Confluent Schema Registry
	@printf "$(BOLD)$(CYAN)== Seeding Avro Schemas → $(SCHEMA_REGISTRY) ==$(RESET)\n"
	@$(KUBECTL) port-forward -n lens-kafka \
		svc/lens-kafka-schema-registry 8081:8081 \
		--address 127.0.0.1 &
	@PF_PID=$$!; sleep 3; \
	for schema_file in $(SCHEMAS_DIR)/*.avsc; do \
		subject=$$(basename $$schema_file .avsc); \
		printf "  Registering $(BOLD)$$subject$(RESET)... "; \
		payload=$$(jq -c '. | {schema: (. | tojson)}' $$schema_file 2>/dev/null || \
			printf '{"schema":"%s"}' "$$(cat $$schema_file | sed 's/"/\\"/g' | tr -d '\n')"); \
		http_code=$$(curl -s -o /tmp/schema_reg_resp.json -w "%{http_code}" \
			-X POST "$(SCHEMA_REGISTRY)/subjects/$$subject-value/versions" \
			-H "Content-Type: application/vnd.schemaregistry.v1+json" \
			-d "$$payload"); \
		if [ "$$http_code" = "200" ] || [ "$$http_code" = "201" ]; then \
			id=$$(jq -r '.id' /tmp/schema_reg_resp.json); \
			printf "$(GREEN)OK (id=$$id)$(RESET)\n"; \
		else \
			printf "$(RED)FAIL (HTTP $$http_code)$(RESET)\n"; \
			cat /tmp/schema_reg_resp.json; \
		fi; \
	done; \
	kill $$PF_PID 2>/dev/null
	@printf "\n$(GREEN)Schema seeding complete.$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
rotate-secrets: ## Rotate credentials for MinIO, PostgreSQL, Redis, Kafka (runs scripts/rotate-secrets.sh)
	@printf "$(BOLD)$(CYAN)== Secret Rotation [env=$(ENV)] ==$(RESET)\n"
	@printf "$(YELLOW)This will generate new credentials and update Kubernetes Secrets.$(RESET)\n"
	@read -p "Proceed? [y/N]: " confirm && [ "$$confirm" = "y" ] || { echo "Aborted."; exit 1; }
	@bash $(SCRIPTS_DIR)/rotate-secrets.sh $(ENV)
	@printf "$(GREEN)Secret rotation complete. Verify pods have restarted successfully.$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
clean: ## Remove Helm cache, temp files, and PID files
	@rm -f /tmp/pf-*.pid /tmp/schema_reg_resp.json
	@find . -name "*.bak" -delete
	@printf "$(GREEN)Clean complete.$(RESET)\n"

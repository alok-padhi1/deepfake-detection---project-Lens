// lib/hooks.ts
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { Case, HealthStatus } from "./types";

export function useCase(caseId: string, token?: string) {
  return useQuery<Case>({
    queryKey: ["case", caseId],
    queryFn: () => api.getCase(caseId, token),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data && (data.status === "PROCESSING" || data.status === "INGESTED")) {
        return 5000; // Poll every 5s during ingest/processing
      }
      return false;
    },
    enabled: !!caseId,
  });
}

export function useCases(filters: Record<string, any>, token?: string) {
  return useQuery<Case[]>({
    queryKey: ["cases", filters],
    queryFn: async () => {
      const resp = await fetch(`/api/v1/cases?${new URLSearchParams(filters)}`, {
        headers: token ? { "Authorization": `Bearer ${token}` } : {},
      });
      return resp.json();
    },
  });
}

export function useHealth() {
  return useQuery<HealthStatus>({
    queryKey: ["health"],
    queryFn: () => api.getHealth(),
    refetchInterval: 30000, // Poll health metrics every 30s
  });
}

export function useSubmitMedia(token?: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ file, metadata }: { file: File; metadata: Record<string, any> }) =>
      api.analyzeMedia(file, metadata, token),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["cases"] });
    },
  });
}

export function useAdminUsers(token?: string) {
  return useQuery<any[]>({
    queryKey: ["admin", "users"],
    queryFn: async () => {
      const resp = await fetch("/api/v1/admin/users", {
        headers: token ? { "Authorization": `Bearer ${token}` } : {},
      });
      return resp.json();
    },
  });
}

export function useAdminModels(token?: string) {
  return useQuery<any[]>({
    queryKey: ["admin", "models"],
    queryFn: async () => {
      const resp = await fetch("/api/v1/admin/models", {
        headers: token ? { "Authorization": `Bearer ${token}` } : {},
      });
      return resp.json();
    },
  });
}

export function useAuditLog(token?: string) {
  return useQuery<any[]>({
    queryKey: ["admin", "audit"],
    queryFn: async () => {
      const resp = await fetch("/api/v1/admin/audit", {
        headers: token ? { "Authorization": `Bearer ${token}` } : {},
      });
      return resp.json();
    },
  });
}

// lib/api.ts
import { Case, HealthStatus } from "./types";

const BASE_URL = "/api/v1";

async function fetchWithAuth(url: string, token?: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers || {});
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  
  const resp = await fetch(`${BASE_URL}${url}`, {
    ...options,
    headers,
  });
  
  if (resp.status === 401) {
    // Session expired
    if (typeof window !== "undefined") {
      window.location.href = "/logout";
    }
    throw new Error("Unauthorized");
  }
  
  if (resp.status === 403) {
    throw new Error("Forbidden");
  }
  
  if (!resp.ok) {
    throw new Error(`API Error: ${resp.statusText}`);
  }
  
  return resp;
}

export const api = {
  async analyzeMedia(file: File, metadata: Record<string, any>, token?: string): Promise<{ case_id: string; status_url: string; estimated_duration_seconds: number }> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("metadata", JSON.stringify(metadata));
    
    const resp = await fetchWithAuth("/analyze", token, {
      method: "POST",
      body: formData,
    });
    return resp.json();
  },

  async getCase(caseId: string, token?: string): Promise<Case> {
    const resp = await fetchWithAuth(`/cases/${caseId}`, token);
    return resp.json();
  },

  async getCaseReport(caseId: string, token?: string): Promise<Blob> {
    const resp = await fetchWithAuth(`/cases/${caseId}/report`, token);
    return resp.blob();
  },

  async getCaseEvidence(caseId: string, token?: string): Promise<Blob> {
    const resp = await fetchWithAuth(`/cases/${caseId}/evidence`, token);
    return resp.blob();
  },

  async getCaseHeatmap(caseId: string, token?: string): Promise<Blob> {
    const resp = await fetchWithAuth(`/cases/${caseId}/heatmap`, token);
    return resp.blob();
  },

  async registerWebhook(url: string, secret: string, token?: string): Promise<{ status: string }> {
    const formData = new FormData();
    formData.append("url", url);
    formData.append("secret", secret);
    
    const resp = await fetchWithAuth("/webhook", token, {
      method: "POST",
      body: formData,
    });
    return resp.json();
  },

  async getHealth(): Promise<HealthStatus> {
    const resp = await fetch(`${BASE_URL}/health`);
    if (!resp.ok) {
      throw new Error("Failed to fetch health status");
    }
    return resp.json();
  }
};

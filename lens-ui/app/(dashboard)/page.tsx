// app/(dashboard)/page.tsx
"use client";
import React from "react";
import { useHealth } from "@/lib/hooks";
import DashboardCharts from "@/components/forensic/DashboardCharts";
import { Cpu, AlertTriangle, Clock, ShieldAlert } from "lucide-react";

export default function DashboardOverview() {
  const { data: health, isLoading } = useHealth();

  const metrics = {
    today: 142,
    week: 852,
    month: 3240,
    gpu: health?.gpu_pool_utilization ? `${(health.gpu_pool_utilization * 100).toFixed(0)}%` : "12%",
    active: health?.kafka_lag !== undefined ? 5 : 3,
  };

  const recentAlerts = [
    { id: "case-941f-82a1", type: "Video", score: 0.94, time: "2 mins ago" },
    { id: "case-019d-fa83", type: "Audio", score: 0.81, time: "12 mins ago" },
    { id: "case-bc42-990a", type: "Image", score: 0.76, time: "24 mins ago" },
    { id: "case-f81d-23a1", type: "Video", score: 0.88, time: "1 hour ago" },
    { id: "case-7a89-bc34", type: "Image", score: 0.91, time: "2 hours ago" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white font-mono">FORENSIC OBSERVABILITY LEDGER</h1>
          <p className="text-xs text-gray-400">Real-time deepfake analysis pipeline and GPU cluster instrumentation</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 rounded bg-green-500/10 border border-green-500/20 text-green-500 text-xs font-mono">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          GATEWAY SECURE
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        
        <div className="bg-surface border border-border p-5 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[10px] uppercase text-gray-500 tracking-wider">Cases In Pipeline</span>
            <div className="text-2xl font-bold text-white font-mono mt-1">{metrics.active}</div>
          </div>
          <AlertTriangle size={24} className="text-forensic-amber" />
        </div>

        <div className="bg-surface border border-border p-5 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[10px] uppercase text-gray-500 tracking-wider">GPU Cluster Load</span>
            <div className="text-2xl font-bold text-white font-mono mt-1">{metrics.gpu}</div>
          </div>
          <Cpu size={24} className="text-forensic-blue" />
        </div>

        <div className="bg-surface border border-border p-5 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[10px] uppercase text-gray-500 tracking-wider">Cases Processed (Today)</span>
            <div className="text-2xl font-bold text-white font-mono mt-1">{metrics.today}</div>
          </div>
          <ShieldAlert size={24} className="text-forensic-green" />
        </div>

        <div className="bg-surface border border-border p-5 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[10px] uppercase text-gray-500 tracking-wider">Avg Latency (Video)</span>
            <div className="text-2xl font-bold text-white font-mono mt-1">14.8s</div>
          </div>
          <Clock size={24} className="text-gray-400" />
        </div>

      </div>

      <DashboardCharts syntheticCount={242} authenticCount={610} />

      {/* Recent Alerts */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <h3 className="text-sm font-semibold uppercase tracking-wider mb-4 font-mono text-gray-300">
          🚨 Active Synthesization Alerts (Confidence &ge; 0.71)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-border text-gray-500">
                <th className="py-2.5 font-mono">CASE ID</th>
                <th className="py-2.5">MEDIA TYPE</th>
                <th className="py-2.5">CONFIDENCE</th>
                <th className="py-2.5">SEVERITY</th>
                <th className="py-2.5">TIMESTAMP</th>
              </tr>
            </thead>
            <tbody>
              {recentAlerts.map((alert) => (
                <tr key={alert.id} className="border-b border-border/50 hover:bg-white/5 font-mono">
                  <td className="py-3 text-forensic-blue">{alert.id}</td>
                  <td className="py-3 text-gray-300">{alert.type}</td>
                  <td className="py-3 font-bold text-forensic-danger">{(alert.score * 100).toFixed(0)}%</td>
                  <td className="py-3">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${alert.score > 0.90 ? "bg-red-950 border border-red-700/30 text-red-500" : "bg-orange-950 border border-orange-700/30 text-orange-500"}`}>
                      {alert.score > 0.90 ? "CRITICAL" : "HIGH"}
                    </span>
                  </td>
                  <td className="py-3 text-gray-500">{alert.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}

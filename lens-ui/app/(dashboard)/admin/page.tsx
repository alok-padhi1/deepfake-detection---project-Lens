// app/(dashboard)/admin/page.tsx
"use client";
import React from "react";
import { Cpu, Server, Layers, ShieldCheck, HelpCircle } from "lucide-react";

export default function AdminOverviewPage() {
  const healthCards = [
    { name: "Image Forensics Service", status: "HEALTHY", load: "12% CPU", color: "text-green-500" },
    { name: "Temporal Video Forensics", status: "HEALTHY", load: "84% GPU (A100)", color: "text-green-500" },
    { name: "Audio Forensics Service", status: "HEALTHY", load: "42% CPU", color: "text-green-500" },
    { name: "AV Synchrony Service", status: "HEALTHY", load: "24% GPU", color: "text-green-500" },
    { name: "Bayesian Fusion Layer", status: "HEALTHY", load: "2% CPU", color: "text-green-500" },
    { name: "API Gateway Proxy", status: "HEALTHY", load: "8% CPU", color: "text-green-500" }
  ];

  return (
    <div className="space-y-6">
      
      {/* 1. Core Health Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        <div className="bg-surface border border-border p-5 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[10px] uppercase text-gray-500 tracking-wider">Active K8s Worker Nodes</span>
            <div className="text-2xl font-bold text-white font-mono mt-1">14 / 14</div>
          </div>
          <Server size={24} className="text-forensic-blue" />
        </div>

        <div className="bg-surface border border-border p-5 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[10px] uppercase text-gray-500 tracking-wider">KEDA Kafka Ingest Lag</span>
            <div className="text-2xl font-bold text-white font-mono mt-1">0 msg</div>
          </div>
          <Layers size={24} className="text-forensic-green" />
        </div>

        <div className="bg-surface border border-border p-5 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[10px] uppercase text-gray-500 tracking-wider">PG Replica Latency</span>
            <div className="text-2xl font-bold text-white font-mono mt-1">0.12 ms</div>
          </div>
          <Cpu size={24} className="text-forensic-blue" />
        </div>

      </div>

      {/* 2. Microservice Node Telemetry Status */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <h3 className="text-xs font-semibold uppercase tracking-wider mb-4 font-mono text-gray-300">
          ⚙️ Forensic Cluster Worker Node Health Matrix
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {healthCards.map((srv) => (
            <div key={srv.name} className="border border-border p-4 rounded-lg bg-black/10 flex items-center justify-between font-mono">
              <div>
                <div className="text-xs font-bold text-white">{srv.name}</div>
                <div className="text-[10px] text-gray-500 mt-1 uppercase">CURRENT LOAD: {srv.load}</div>
              </div>
              <span className={`text-xs font-extrabold ${srv.color}`}>{srv.status}</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}

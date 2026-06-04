// app/(dashboard)/admin/models/page.tsx
"use client";
import React, { useState } from "react";
import { Sliders, RefreshCcw, Activity, AlertTriangle } from "lucide-react";

export default function AdminModelsPage() {
  const [models, setModels] = useState([
    { name: "EfficientNet-B0 (Image)", version: "b0-v1.2", auc: 0.9934, deployed: "2026-05-16", status: "ACTIVE", latency: "14ms", drift: 0.02 },
    { name: "TimeSformer (Video Temporal)", version: "v2.1", auc: 0.9780, deployed: "2026-05-16", status: "ACTIVE", latency: "112ms", drift: 0.09 },
    { name: "ResNet-34 (Audio Spectrogram)", version: "audio-v1", auc: 0.9650, deployed: "2026-05-15", status: "ACTIVE", latency: "22ms", drift: 0.03 },
    { name: "RawNet3 (Audio Waveform)", version: "v3", auc: 0.9580, deployed: "2026-05-15", status: "ACTIVE", latency: "48ms", drift: 0.01 },
    { name: "SyncNet (AV Synchrony)", version: "v1.1", auc: 0.9420, deployed: "2026-05-17", status: "ACTIVE", latency: "90ms", drift: 0.04 },
    { name: "Bayesian MLP (Fusion)", version: "v1", auc: 0.9910, deployed: "2026-05-17", status: "ACTIVE", latency: "2ms", drift: 0.005 }
  ]);

  const handleRetrain = (name: string) => {
    setModels(prev => prev.map(m => m.name === name ? { ...m, status: "RETRAINING" } : m));
    setTimeout(() => {
      setModels(prev => prev.map(m => m.name === name ? { ...m, status: "ACTIVE", auc: parseFloat((m.auc + 0.002).toFixed(4)) } : m));
    }, 3000);
  };

  // Check if any model exceeds drift boundaries
  const driftedModel = models.find(m => m.drift > 0.08);

  return (
    <div className="space-y-6">
      
      {driftedModel && (
        <div className="bg-orange-950 border border-orange-500 text-orange-400 p-4 rounded-xl flex items-center gap-3 font-mono text-xs">
          <AlertTriangle size={20} className="text-orange-500 animate-pulse" />
          <div>
            <span className="font-extrabold text-sm block">MODEL PERFORMANCE DRIFT DETECTED</span>
            {driftedModel.name} exhibits drift coefficient: {driftedModel.drift} (Boundary threshold: 0.08).
          </div>
        </div>
      )}

      {/* Model registry list */}
      <div className="bg-surface border border-border rounded-xl p-5 space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider font-mono text-gray-300 flex items-center gap-1.5">
          <Activity size={16} className="text-forensic-blue" /> Deployed Deepfake Detection Model Weights
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-gray-500 uppercase tracking-wider">
                <th className="py-2.5">MODEL NAME</th>
                <th className="py-2.5">WEIGHT VERSION</th>
                <th className="py-2.5">TEST AUC</th>
                <th className="py-2.5">p50 LATENCY</th>
                <th className="py-2.5">DRIFT VALUE</th>
                <th className="py-2.5">STATUS</th>
                <th className="py-2.5">DEPLOYED DATE</th>
                <th className="py-2.5 text-center">MAINTENANCE</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name} className="border-b border-border/40 hover:bg-white/5">
                  <td className="py-3 text-white font-bold">{m.name}</td>
                  <td className="py-3 text-gray-400">{m.version}</td>
                  <td className="py-3 text-forensic-green font-bold">{(m.auc * 100).toFixed(2)}%</td>
                  <td className="py-3 text-gray-300">{m.latency}</td>
                  <td className={`py-3 font-bold ${m.drift > 0.08 ? "text-forensic-amber" : "text-gray-500"}`}>{m.drift}</td>
                  <td className="py-3">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${m.status === "ACTIVE" ? "bg-green-500/10 text-green-500" : "bg-blue-500/10 text-forensic-blue animate-pulse"}`}>
                      {m.status}
                    </span>
                  </td>
                  <td className="py-3 text-gray-500">{m.deployed}</td>
                  <td className="py-3 text-center">
                    <button 
                      onClick={() => handleRetrain(m.name)}
                      disabled={m.status === "RETRAINING"}
                      className="inline-flex items-center gap-1 bg-forensic-blue/10 hover:bg-forensic-blue/20 text-forensic-blue px-2.5 py-1 rounded text-[10px] font-bold border border-forensic-blue/20 transition-all"
                    >
                      <RefreshCcw size={10} className={m.status === "RETRAINING" ? "animate-spin" : ""} />
                      Trigger Retraining
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}

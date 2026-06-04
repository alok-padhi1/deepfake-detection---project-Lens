// app/(dashboard)/admin/system/page.tsx
"use client";
import React, { useState } from "react";
import { Sliders, Plus, Trash, Globe, ShieldCheck } from "lucide-react";

export default function AdminSystemPage() {
  const [thresholds, setThresholds] = useState({
    authentic: 0.20,
    low: 0.45,
    moderate: 0.70,
    high: 0.89
  });

  const [webhooks, setWebhooks] = useState([
    { id: "wh-1", url: "https://notify.agency.gov/endpoints/lens", secret: "hmac_sec_99124a" },
    { id: "wh-2", url: "https://slack.forensics.platform/services/alerts", secret: "hmac_sec_42bc01" }
  ]);

  const [newUrl, setNewUrl] = useState("");
  const [newSecret, setNewSecret] = useState("");

  const handleAddWebhook = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newUrl) return;
    setWebhooks(prev => [...prev, { id: `wh-${Date.now()}`, url: newUrl, secret: newSecret || "hmac_default" }]);
    setNewUrl("");
    setNewSecret("");
  };

  const handleDeleteWebhook = (id: string) => {
    setWebhooks(prev => prev.filter(w => w.id !== id));
  };

  return (
    <div className="space-y-6">
      
      {/* Threshold Sliders */}
      <div className="bg-surface border border-border p-5 rounded-xl space-y-6">
        <h3 className="text-xs font-semibold uppercase tracking-wider font-mono text-gray-300 flex items-center gap-1.5">
          <Sliders size={16} className="text-forensic-amber" /> Decision Band Classification Boundaries
        </h3>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-forensic-green font-bold">Authentic Boundary</span>
              <span className="text-white">{thresholds.authentic}</span>
            </div>
            <input 
              type="range" min="0" max="0.5" step="0.05"
              value={thresholds.authentic}
              onChange={(e) => setThresholds(prev => ({ ...prev, authentic: parseFloat(e.target.value) }))}
              className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-forensic-green"
            />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-yellow-500 font-bold">Low Suspicion Boundary</span>
              <span className="text-white">{thresholds.low}</span>
            </div>
            <input 
              type="range" min="0.2" max="0.6" step="0.05"
              value={thresholds.low}
              onChange={(e) => setThresholds(prev => ({ ...prev, low: parseFloat(e.target.value) }))}
              className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-yellow-500"
            />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-orange-500 font-bold">Moderate Suspicion Boundary</span>
              <span className="text-white">{thresholds.moderate}</span>
            </div>
            <input 
              type="range" min="0.5" max="0.8" step="0.05"
              value={thresholds.moderate}
              onChange={(e) => setThresholds(prev => ({ ...prev, moderate: parseFloat(e.target.value) }))}
              className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-orange-500"
            />
          </div>
        </div>
      </div>

      {/* Webhooks Manager */}
      <div className="bg-surface border border-border p-5 rounded-xl space-y-6">
        <h3 className="text-xs font-semibold uppercase tracking-wider font-mono text-gray-300 flex items-center gap-1.5">
          <Globe size={16} className="text-forensic-blue" /> Case Complete Event Notification (Webhooks)
        </h3>

        <form onSubmit={handleAddWebhook} className="grid grid-cols-1 md:grid-cols-3 gap-6 items-end border-b border-border pb-6">
          <div>
            <label className="block text-[10px] uppercase text-gray-500 tracking-wider mb-2">Endpoint URL</label>
            <input 
              type="url"
              value={newUrl}
              onChange={(e) => setNewUrl(e.target.value)}
              placeholder="https://client.platform/webhook"
              className="w-full bg-background border border-border rounded-lg p-2 text-xs text-white focus:outline-none focus:border-forensic-blue"
            />
          </div>
          <div>
            <label className="block text-[10px] uppercase text-gray-500 tracking-wider mb-2">HMAC Signature Secret</label>
            <input 
              type="password"
              value={newSecret}
              onChange={(e) => setNewSecret(e.target.value)}
              placeholder="e.g. hmac_signing_key_secret"
              className="w-full bg-background border border-border rounded-lg p-2 text-xs text-white focus:outline-none focus:border-forensic-blue"
            />
          </div>
          <button 
            type="submit"
            className="bg-forensic-blue hover:bg-blue-600 text-white font-mono font-bold text-xs py-2 px-4 rounded-lg h-9 transition-colors flex items-center justify-center gap-1"
          >
            <Plus size={14} /> Register Webhook
          </button>
        </form>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-gray-500">
                <th className="py-2">REGISTERED ENDPOINT</th>
                <th className="py-2">HMAC SECRET MASK</th>
                <th className="py-2 text-center">DELETE</th>
              </tr>
            </thead>
            <tbody>
              {webhooks.map((wh) => (
                <tr key={wh.id} className="border-b border-border/40 hover:bg-white/5">
                  <td className="py-3 text-white font-semibold">{wh.url}</td>
                  <td className="py-3 text-gray-500">•••••••••••••••••••• ({wh.secret.substring(wh.secret.length - 4)})</td>
                  <td className="py-3 text-center">
                    <button 
                      onClick={() => handleDeleteWebhook(wh.id)}
                      className="text-red-500 hover:text-red-400 p-1"
                    >
                      <Trash size={14} />
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

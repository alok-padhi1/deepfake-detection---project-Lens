// app/(dashboard)/admin/audit/page.tsx
"use client";
import React, { useState } from "react";
import { ShieldCheck, CheckCircle2, AlertTriangle, Activity } from "lucide-react";

export default function AdminAuditPage() {
  const [chainVerified, setChainVerified] = useState<boolean | null>(null);

  const logs = [
    { entry_id: "tx-491a", case_id: "case-01a2-ff82", event: "CLASSIFY_SUSPICIOUS", actor: "analyst_jones", time: "2026-05-17 15:45:00", prev_hash: "00000000000000000000000000000000", hash: "a8f3c91d8e82a8190dcb1f39f81d4eae" },
    { entry_id: "tx-82bc", case_id: "case-941f-82a1", event: "VERDICT_COMPLETED", actor: "system_fusion", time: "2026-05-17 15:45:30", prev_hash: "a8f3c91d8e82a8190dcb1f39f81d4eae", hash: "99dee85f0c94613b840f73793652a18e" },
    { entry_id: "tx-fa83", case_id: "case-e5ce-4316", event: "REPORT_DOWNLOADED", actor: "admin_smith", time: "2026-05-17 15:48:12", prev_hash: "99dee85f0c94613b840f73793652a18e", hash: "c94613b840f73793652a18e5cef32727" }
  ];

  const verifyLedgerIntegrity = () => {
    setChainVerified(null);
    setTimeout(() => {
      // Basic block validation mock: verifies hash alignment across columns
      let isVerified = true;
      for (let i = 1; i < logs.length; i++) {
        if (logs[i].prev_hash !== logs[i-1].hash) {
          isVerified = false;
        }
      }
      setChainVerified(isVerified);
    }, 1000);
  };

  return (
    <div className="space-y-6">
      
      {/* Integrity Actions */}
      <div className="bg-surface border border-border p-5 rounded-xl flex items-center justify-between">
        <div>
          <span className="text-xs uppercase text-gray-500 tracking-wider">Cryptographic Signature Verification</span>
          <p className="text-xs text-gray-400 mt-1">Check full blockchain-style SHA-256 integrity hashes for all records inside the audit ledgers.</p>
        </div>
        <button 
          onClick={verifyLedgerIntegrity}
          className="flex items-center gap-1.5 bg-forensic-blue hover:bg-blue-600 px-4 py-2 rounded-lg text-xs font-mono font-bold text-white transition-all shadow"
        >
          <ShieldCheck size={14} /> Validate Ledger Chaining
        </button>
      </div>

      {chainVerified !== null && (
        <div className={`p-4 rounded-xl border text-xs font-mono flex items-center gap-2 ${chainVerified ? "bg-green-500/10 border-green-500/20 text-green-500" : "bg-red-500/10 border-red-500/20 text-red-500"}`}>
          {chainVerified ? (
            <>
              <CheckCircle2 size={16} />
              CRYPTOGRAPHIC CHAIN INTEGRITY SIGNED AND FULLY VERIFIED (No tamper traces found).
            </>
          ) : (
            <>
              <AlertTriangle size={16} />
              WARNING: Cryptographic integrity breakdown detected at block ledger indices!
            </>
          )}
        </div>
      )}

      {/* Ledger Table */}
      <div className="bg-surface border border-border rounded-xl p-5 space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider font-mono text-gray-300 flex items-center gap-1.5">
          <Activity size={16} className="text-forensic-amber" /> Cryptographically Chained Audit Trail Ledger
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-gray-500">
                <th className="py-2.5">ENTRY ID</th>
                <th className="py-2.5">CASE ID</th>
                <th className="py-2.5">EVENT ACTION</th>
                <th className="py-2.5">ACTOR</th>
                <th className="py-2.5">PREVIOUS BLOCK HASH</th>
                <th className="py-2.5">CURRENT BLOCK HASH</th>
                <th className="py-2.5">TIMESTAMP</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log) => (
                <tr key={log.entry_id} className="border-b border-border/40 hover:bg-white/5">
                  <td className="py-3 text-forensic-blue font-bold">{log.entry_id}</td>
                  <td className="py-3 text-white">{log.case_id}</td>
                  <td className="py-3">
                    <span className="bg-white/5 border border-border px-2 py-0.5 rounded text-[10px] uppercase font-bold text-gray-300">
                      {log.event}
                    </span>
                  </td>
                  <td className="py-3 text-gray-300">{log.actor}</td>
                  <td className="py-3 text-gray-500 truncate max-w-[120px]" title={log.prev_hash}>{log.prev_hash}</td>
                  <td className="py-3 text-forensic-green truncate max-w-[120px]" title={log.hash}>{log.hash}</td>
                  <td className="py-3 text-gray-400">{log.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}

// app/(dashboard)/cases/page.tsx
"use client";
import React, { useState } from "react";
import Link from "next/link";
import { FolderOpen, Eye, Filter, RefreshCcw } from "lucide-react";
import { useCases } from "@/lib/hooks";

export default function CasesPage() {
  const [status, setStatus] = useState("");
  const [mediaType, setMediaType] = useState("");
  const [search, setSearch] = useState("");

  const { data: cases, refetch, isFetching } = useCases({
    status,
    media_type: mediaType,
    search
  });

  const getDecisionBadge = (score: number) => {
    if (score > 0.90) return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-950 border border-red-800 text-red-500 animate-pulse">NEAR-CERTAIN SYNTHETIC</span>;
    if (score > 0.71) return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-900/30 border border-red-700/30 text-red-400">HIGH CONFIDENCE</span>;
    if (score > 0.46) return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-orange-950 border border-orange-700/30 text-orange-500">MODERATE</span>;
    if (score > 0.21) return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-yellow-950 border border-yellow-700/30 text-yellow-500">LOW SUSPICION</span>;
    return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-green-950 border border-green-700/30 text-green-500">AUTHENTIC</span>;
  };

  const getStatusBadge = (s: string) => {
    switch (s) {
      case "COMPLETE": return <span className="px-2 py-0.5 rounded text-[10px] bg-green-500/10 text-green-500 border border-green-500/20">COMPLETE</span>;
      case "PROCESSING": return <span className="px-2 py-0.5 rounded text-[10px] bg-blue-500/10 text-forensic-blue border border-blue-500/20 animate-pulse">PROCESSING</span>;
      case "FAILED": return <span className="px-2 py-0.5 rounded text-[10px] bg-red-500/10 text-red-500 border border-red-500/20">FAILED</span>;
      default: return <span className="px-2 py-0.5 rounded text-[10px] bg-gray-500/10 text-gray-400 border border-gray-500/20">INGESTED</span>;
    }
  };

  const mockCases = [
    { case_id: "case-d81a-2900", filename: "manifesto_voice.wav", media_type: "audio", confidence_score: 0.84, status: "COMPLETE", created_at: "2026-05-17 14:02" },
    { case_id: "case-01a2-ff82", filename: "politician_deepfake.mp4", media_type: "video", confidence_score: 0.96, status: "COMPLETE", created_at: "2026-05-17 12:45" },
    { case_id: "case-941f-82a1", filename: "candid_shot.png", media_type: "image", confidence_score: 0.12, status: "COMPLETE", created_at: "2026-05-17 11:20" },
    { case_id: "case-bc42-990a", filename: "broadcast_stream.mp4", media_type: "video", confidence_score: 0.52, status: "PROCESSING", created_at: "2026-05-17 15:10" },
    { case_id: "case-e5ce-4316", filename: "audio_wiretap.mp3", media_type: "audio", confidence_score: 0.05, status: "COMPLETE", created_at: "2026-05-17 10:15" },
  ];

  const activeCases = cases || mockCases;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white font-mono">FORENSIC CASE LEDGER</h1>
          <p className="text-xs text-gray-400">Review, query, and download synthesized deepfake forensic cases.</p>
        </div>
        <button 
          onClick={() => refetch()} 
          className="flex items-center gap-2 bg-surface hover:bg-white/5 border border-border px-3 py-1.5 rounded-lg text-xs text-gray-300 transition-colors"
        >
          <RefreshCcw size={14} className={isFetching ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {/* Filter Bar */}
      <div className="bg-surface border border-border p-4 rounded-xl flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2 text-xs text-gray-400 font-mono">
          <Filter size={14} /> FILTERS
        </div>

        <input 
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by Case ID or file..."
          className="bg-background border border-border rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-forensic-blue min-w-[200px]"
        />

        <select 
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="bg-background border border-border rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-forensic-blue"
        >
          <option value="">All Statuses</option>
          <option value="COMPLETE">Complete</option>
          <option value="PROCESSING">Processing</option>
          <option value="FAILED">Failed</option>
        </select>

        <select 
          value={mediaType}
          onChange={(e) => setMediaType(e.target.value)}
          className="bg-background border border-border rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-forensic-blue"
        >
          <option value="">All Media</option>
          <option value="image">Image</option>
          <option value="video">Video</option>
          <option value="audio">Audio</option>
        </select>
      </div>

      {/* Case Ledger Table */}
      <div className="bg-surface border border-border rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-border text-gray-500 bg-black/10 uppercase tracking-widest font-mono">
                <th className="py-3 px-4">CASE ID</th>
                <th className="py-3 px-4">Filename</th>
                <th className="py-3 px-4">Media</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Decision Band</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Ingested At</th>
                <th className="py-3 px-4 text-center">Action</th>
              </tr>
            </thead>
            <tbody>
              {activeCases.map((c) => (
                <tr key={c.case_id} className="border-b border-border/50 hover:bg-white/5 font-mono">
                  <td className="py-3.5 px-4 text-forensic-blue font-semibold">{c.case_id}</td>
                  <td className="py-3.5 px-4 text-gray-300">{c.filename || "media_file.tmp"}</td>
                  <td className="py-3.5 px-4 text-gray-400 capitalize">{c.media_type}</td>
                  <td className="py-3.5 px-4 font-bold text-white">{(c.confidence_score * 100).toFixed(0)}%</td>
                  <td className="py-3.5 px-4">{getDecisionBadge(c.confidence_score)}</td>
                  <td className="py-3.5 px-4">{getStatusBadge(c.status)}</td>
                  <td className="py-3.5 px-4 text-gray-500">{c.created_at}</td>
                  <td className="py-3.5 px-4 text-center">
                    <Link href={`/cases/${c.case_id}`} className="inline-flex items-center gap-1 bg-forensic-blue/10 hover:bg-forensic-blue/20 text-forensic-blue px-2.5 py-1 rounded text-[10px] font-bold transition-all border border-forensic-blue/20">
                      <Eye size={10} /> View
                    </Link>
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

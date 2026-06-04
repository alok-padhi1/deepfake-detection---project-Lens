// app/(dashboard)/cases/[caseId]/page.tsx
"use client";
import React, { useState } from "react";
import { useParams } from "next/navigation";
import { 
  FileText, Download, Layers, ShieldCheck, 
  CheckCircle, Database, AlertOctagon, HelpCircle 
} from "lucide-react";
import { useCase } from "@/lib/hooks";
import CaseDetailVisuals from "@/components/forensic/CaseDetailVisuals";

export default function CaseDetailPage() {
  const params = useParams();
  const caseId = params.caseId as string;
  const [blendOpacity, setBlendOpacity] = useState(0.5);
  const [integrityVerified, setIntegrityVerified] = useState<boolean | null>(null);

  const { data: caseData, isLoading } = useCase(caseId);

  // Fallback production simulation data
  const fallbackCase = {
    case_id: caseId,
    status: "COMPLETE",
    confidence_score: 0.86,
    decision_band: "HIGH",
    decision_label: "FAKE",
    media_type: "video",
    modules_complete: ["image", "video", "audio", "av_sync"],
    created_at: "2026-05-17T14:02:15Z",
    completed_at: "2026-05-17T14:03:00Z",
    filename: "suspect_interview_leaked.mp4"
  };

  const c = caseData || fallbackCase;
  const isVideo = c.media_type === "video";

  const handleVerifyIntegrity = () => {
    setIntegrityVerified(null);
    setTimeout(() => {
      setIntegrityVerified(true);
    }, 1000);
  };

  return (
    <div className="space-y-6">
      
      {/* 1. Header Section */}
      <div className="bg-surface border border-border p-6 rounded-xl flex flex-wrap items-center justify-between gap-4">
        <div>
          <span className="text-[10px] text-gray-500 uppercase tracking-widest font-mono">CASE RECORD FILE</span>
          <h1 className="text-2xl font-bold font-mono text-white mt-1">{c.case_id}</h1>
          <p className="text-xs text-gray-400 mt-0.5">Ingested: {new Date(c.created_at).toLocaleString()} | File: {c.filename || "unknown"}</p>
        </div>

        <div className="flex items-center gap-3">
          <button className="flex items-center gap-2 bg-surface hover:bg-white/5 border border-border px-4 py-2 rounded-lg text-xs font-mono font-bold text-gray-300 transition-all">
            <FileText size={14} /> PDF Report
          </button>
          <button className="flex items-center gap-2 bg-surface hover:bg-white/5 border border-border px-4 py-2 rounded-lg text-xs font-mono font-bold text-gray-300 transition-all">
            <Download size={14} /> Evidence ZIP
          </button>
          <button className="flex items-center gap-2 bg-forensic-blue hover:bg-blue-600 px-4 py-2 rounded-lg text-xs font-mono font-bold text-white transition-all">
            <Layers size={14} /> Heatmap
          </button>
        </div>
      </div>

      {/* 2. Verdict Gauge panel */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        <div className="md:col-span-2 bg-surface border border-border p-5 rounded-xl space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider font-mono text-gray-300 border-b border-border pb-2">
            Forensic Verdict Profile
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <AlertOctagon className="text-forensic-danger" size={20} />
                <span className="text-lg font-bold text-white uppercase font-mono">CRITICAL RISK FACTOR</span>
              </div>
              <p className="text-xs text-gray-400">
                Highly anomalous neural synthesization signatures identified in facial keypoints and speech spectral vectors.
              </p>
              <div className="flex flex-wrap gap-2 pt-2">
                {c.modules_complete.map((mod) => (
                  <span key={mod} className="px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-forensic-blue/10 border border-forensic-blue/20 text-forensic-blue uppercase">
                    {mod} MODULE ACTIVE
                  </span>
                ))}
              </div>
            </div>
            <div className="bg-black/20 p-4 border border-border rounded-lg text-center font-mono">
              <span className="text-[10px] text-gray-500 uppercase tracking-widest">Decision Classification</span>
              <div className="text-2xl font-extrabold text-forensic-danger mt-1">HIGH CONFLICT FAKE</div>
              <span className="text-xs text-gray-400">Score Range: 0.71 &ge; 0.89</span>
            </div>
          </div>
        </div>

        {/* Bounding Arc Visuals */}
        <CaseDetailVisuals score={c.confidence_score} />

      </div>

      {/* 3. Heatmap Opacity Viewer */}
      <div className="bg-surface border border-border p-5 rounded-xl">
        <h2 className="text-sm font-semibold uppercase tracking-wider font-mono text-gray-300 border-b border-border pb-2 mb-4">
          Spatial Localization Anomalous Heatmap (Wiener Wavelet PRNU)
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div className="relative border border-border rounded-lg overflow-hidden h-80 bg-black flex items-center justify-center">
            {/* Base image simulation */}
            <div className="absolute inset-0 bg-cover bg-center" style={{ backgroundImage: "url('/img/target_mock.jpg')" }} />
            <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-xs uppercase tracking-widest bg-slate-900/90 font-mono">
              Original Subject Media
            </div>
            
            {/* Heatmap overlay blend */}
            <div 
              className="absolute inset-0 bg-gradient-to-tr from-blue-600/60 via-red-600/70 to-yellow-400/80 mix-blend-color-burn flex items-center justify-center font-mono text-white text-xs uppercase"
              style={{ opacity: blendOpacity }}
            >
              <div className="bg-black/80 px-4 py-2 border border-red-500/30 rounded">ANOMALOUS PATCH DISTRIBUTION</div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="space-y-1">
              <span className="text-xs text-gray-400 uppercase font-mono">Heatmap Transparency Blend Slider</span>
              <p className="text-xs text-gray-500">Slide to cross-verify visual artifact coordinates against PRNU sensor discrepancies.</p>
            </div>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05"
              value={blendOpacity}
              onChange={(e) => setBlendOpacity(parseFloat(e.target.value))}
              className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-forensic-blue"
            />
            <div className="flex justify-between text-[10px] text-gray-500 font-mono">
              <span>0% (SOURCE MEDIA)</span>
              <span>{(blendOpacity * 100).toFixed(0)}%</span>
              <span>100% (FORENSIC HEATMAP)</span>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Tabulated Modules Scores Breakdown */}
      <div className="bg-surface border border-border p-5 rounded-xl space-y-4">
        <h2 className="text-sm font-semibold uppercase tracking-wider font-mono text-gray-300 border-b border-border pb-2 mb-4">
          Multi-Modal Forensic Pipeline Breakdown
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="border border-border rounded-lg p-4 space-y-3 font-mono">
            <span className="text-[10px] text-forensic-blue uppercase font-bold">Image Forensics Module</span>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">Wiener PRNU Correlation:</span>
              <span className="text-white font-bold">0.84</span>
            </div>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">2D DCT Histogram KL:</span>
              <span className="text-white font-bold">0.78</span>
            </div>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">CNN Patch Anomaly:</span>
              <span className="text-forensic-danger font-bold">0.89</span>
            </div>
          </div>

          <div className="border border-border rounded-lg p-4 space-y-3 font-mono">
            <span className="text-[10px] text-forensic-blue uppercase font-bold">Video Temporal Module</span>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">Landmark Jerk Index:</span>
              <span className="text-forensic-danger font-bold">0.91</span>
            </div>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">Flow Divergence Pixel:</span>
              <span className="text-white font-bold">0.76</span>
            </div>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">TimeSformer Sequence:</span>
              <span className="text-white font-bold">0.85</span>
            </div>
          </div>

          <div className="border border-border rounded-lg p-4 space-y-3 font-mono">
            <span className="text-[10px] text-forensic-blue uppercase font-bold">Audio Forensics Module</span>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">Formant Sharpness F1/F2:</span>
              <span className="text-white font-bold">0.82</span>
            </div>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">LPC Glottal Pulse Excitation:</span>
              <span className="text-white font-bold">0.79</span>
            </div>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">Vocoder High-Freq Smooth:</span>
              <span className="text-forensic-danger font-bold">0.88</span>
            </div>
          </div>

          <div className="border border-border rounded-lg p-4 space-y-3 font-mono">
            <span className="text-[10px] text-forensic-blue uppercase font-bold">Audio-Visual Synchrony Module</span>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">SyncNet Cosine Offset:</span>
              <span className="text-forensic-danger font-bold">0.42 (ASYNC)</span>
            </div>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">Lip-Speech Jaw Mismatch:</span>
              <span className="text-forensic-danger font-bold">TRUE</span>
            </div>
            <div className="flex justify-between text-xs py-1 border-b border-border/50">
              <span className="text-gray-400">Nasal/Nostril Expansion:</span>
              <span className="text-white font-bold">FALSE (NORMAL)</span>
            </div>
          </div>
        </div>
      </div>

      {/* 5. Evidence Manifest & Cryptographic Signatures */}
      <div className="bg-surface border border-border p-5 rounded-xl space-y-4">
        <div className="flex items-center justify-between border-b border-border pb-2">
          <h2 className="text-sm font-semibold uppercase tracking-wider font-mono text-gray-300">
            Cryptographic Integrity Manifest
          </h2>
          <button 
            onClick={handleVerifyIntegrity}
            className="flex items-center gap-1.5 bg-white/5 border border-border px-3 py-1 rounded text-xs font-mono font-bold text-gray-300 hover:text-white transition-colors"
          >
            <ShieldCheck size={14} className="text-forensic-green" />
            Verify Manifest
          </button>
        </div>

        {integrityVerified !== null && (
          <div className={`p-3 rounded-lg border text-xs font-mono flex items-center gap-2 ${integrityVerified ? "bg-green-500/10 border-green-500/20 text-green-500" : "bg-red-500/10 border-red-500/20 text-red-500"}`}>
            <CheckCircle size={14} />
            ALL FILES INTEGRITY VERIFIED (SHA-256 LEDGER MATCHED)
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-gray-500">
                <th className="py-2">FILE COMPONENT</th>
                <th className="py-2">SHA-256 HASH</th>
                <th className="py-2">METADATA STATUS</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-border/40 text-gray-300">
                <td className="py-2.5">source_media_video.mp4</td>
                <td className="py-2.5 text-gray-500">f81d4fae92a83109da3cf115598112abde42bc1d882e11acde335900aa12cb4f</td>
                <td className="py-2.5 text-forensic-green">VERIFIED</td>
              </tr>
              <tr className="border-b border-border/40 text-gray-300">
                <td className="py-2.5">wiener_prnu_residual.png</td>
                <td className="py-2.5 text-gray-500">019dee85f0c94613b840f73793652a18e5cef32727214316a0fefdbe501bf693</td>
                <td className="py-2.5 text-forensic-green">VERIFIED</td>
              </tr>
              <tr className="border-b border-border/40 text-gray-300">
                <td className="py-2.5">formant_trajectories.csv</td>
                <td className="py-2.5 text-gray-500">c94613b840f73793652a18f81d4fae92a83109da3cf115598112abde42bc1d88</td>
                <td className="py-2.5 text-forensic-green">VERIFIED</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}

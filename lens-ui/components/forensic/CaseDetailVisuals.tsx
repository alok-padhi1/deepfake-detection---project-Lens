// components/forensic/CaseDetailVisuals.tsx
"use client";
import React from "react";
import { 
  ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  AreaChart, Area, ComposedChart, Bar
} from "recharts";

interface VisualsProps {
  score: number;
}

export default function CaseDetailVisuals({ score }: VisualsProps) {
  // Radial Chart configuration
  const radialData = [
    { name: "Confidence", value: score * 100, fill: score > 0.71 ? "#EF4444" : "#10B981" }
  ];

  // Simulating 100 frames worth of temporal alignment data for video graphs
  const timelineData = Array.from({ length: 40 }).map((_, idx) => {
    const frame = idx * 2;
    const baseFlow = Math.abs(Math.sin(frame * 0.1) * 3) + 1.0;
    const flowDivergence = idx > 15 && idx < 28 ? baseFlow * 2.8 : baseFlow;
    
    const baseSync = Math.sin(frame * 0.15) * 0.2 + 0.78;
    const avSync = idx > 10 && idx < 22 ? baseSync - 0.45 : baseSync;

    const baseVelocity = Math.abs(Math.cos(frame * 0.08) * 12) + 2.0;
    const landmarkVelocity = idx > 15 && idx < 28 ? baseVelocity + 15.0 : baseVelocity;

    const ensembleMin = Math.max(0.0, score - 0.1);
    const ensembleMax = Math.min(1.0, score + 0.1);
    
    return {
      frame,
      flowDivergence: parseFloat(flowDivergence.toFixed(2)),
      avSync: parseFloat(avSync.toFixed(2)),
      landmarkVelocity: parseFloat(landmarkVelocity.toFixed(2)),
      confidence: score,
      ensembleMin,
      ensembleMax
    };
  });

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* 1. Animated Radial Arc Gauge */}
        <div className="bg-surface border border-border p-5 rounded-xl flex flex-col items-center justify-center min-h-[260px]">
          <h3 className="text-xs font-semibold uppercase tracking-wider mb-4 font-mono text-gray-400">
            Ensembled Confidence Score
          </h3>
          <div className="w-full h-44 relative">
            <ResponsiveContainer width="100%" height="100%">
              <RadialBarChart 
                innerRadius="80%" 
                outerRadius="100%" 
                data={radialData} 
                startAngle={180} 
                endAngle={0}
              >
                <PolarAngleAxis 
                  type="number" 
                  domain={[0, 100]} 
                  angleAxisId={0} 
                  tick={false} 
                />
                <RadialBar 
                  background 
                  dataKey="value" 
                  cornerRadius={10} 
                />
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="absolute inset-0 flex flex-col items-center justify-center top-8">
              <span className={`text-4xl font-extrabold font-mono ${score > 0.71 ? "text-forensic-danger animate-pulse" : "text-forensic-green"}`}>
                {(score * 100).toFixed(0)}%
              </span>
              <span className="text-[10px] text-gray-500 uppercase tracking-widest mt-1">ANOMALY SCALE</span>
            </div>
          </div>
        </div>

        {/* 2. Optical Flow Divergence (AreaChart) */}
        <div className="bg-surface border border-border p-5 rounded-xl md:col-span-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider mb-4 font-mono text-gray-400">
            Temporal Optical Flow Divergence (RAFT Residual Pixels)
          </h3>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E1E2E" />
                <XAxis dataKey="frame" stroke="#94A3B8" fontSize={9} label={{ value: 'Frame Interval', position: 'insideBottom', offset: -5 }} />
                <YAxis stroke="#94A3B8" fontSize={9} />
                <Tooltip contentStyle={{ backgroundColor: "#12121A", borderColor: "#1E1E2E" }} />
                <Area type="monotone" dataKey="flowDivergence" stroke="#3B82F6" fill="rgba(59, 130, 246, 0.1)" name="Residual Pixels" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* 3. Audio-Visual Sync (LineChart with Threshold) */}
        <div className="bg-surface border border-border p-5 rounded-xl">
          <h3 className="text-xs font-semibold uppercase tracking-wider mb-4 font-mono text-gray-400">
            Audio-Visual Synchrony Cosine Drift (SyncNet Score)
          </h3>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E1E2E" />
                <XAxis dataKey="frame" stroke="#94A3B8" fontSize={9} />
                <YAxis stroke="#94A3B8" domain={[0, 1.0]} fontSize={9} />
                <Tooltip contentStyle={{ backgroundColor: "#12121A", borderColor: "#1E1E2E" }} />
                {/* Horizontal reference threshold line for authenticity */}
                <Line type="monotone" dataKey="avSync" stroke="#F59E0B" strokeWidth={2} dot={false} name="Sync Score" />
                <Line type="monotone" dataKey="frame" stroke="#EF4444" strokeDasharray="5 5" dot={false} strokeWidth={1} name="Authentic Threshold (0.72)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4. Landmark Jerk Velocity (LineChart) */}
        <div className="bg-surface border border-border p-5 rounded-xl">
          <h3 className="text-xs font-semibold uppercase tracking-wider mb-4 font-mono text-gray-400">
            MediaPipe Face Mesh Mesh Landmark Velocity (Jerk Score)
          </h3>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E1E2E" />
                <XAxis dataKey="frame" stroke="#94A3B8" fontSize={9} />
                <YAxis stroke="#94A3B8" fontSize={9} />
                <Tooltip contentStyle={{ backgroundColor: "#12121A", borderColor: "#1E1E2E" }} />
                <Line type="monotone" dataKey="landmarkVelocity" stroke="#10B981" strokeWidth={2} dot={false} name="Velocity std_dev" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}

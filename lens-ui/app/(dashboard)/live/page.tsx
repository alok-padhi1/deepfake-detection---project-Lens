// app/(dashboard)/live/page.tsx
"use client";
import React, { useState, useEffect, useRef } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { Play, Square, Video, AlertTriangle, Settings, RefreshCw } from "lucide-react";
import { io, Socket } from "socket.io-client";

interface FrameScore {
  frame_number: number;
  confidence: number;
  anomaly_flags: string[];
  timestamp: string;
}

export default function LiveStreamPage() {
  const [streamUrl, setStreamUrl] = useState("rtsp://internal.lens.platform:8554/live/call_12");
  const [isStreaming, setIsStreaming] = useState(false);
  const [frames, setFrames] = useState<FrameScore[]>([]);
  const [consecutiveViolations, setConsecutiveViolations] = useState(0);
  const [lastViolationFrame, setLastViolationFrame] = useState<number | null>(null);

  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    if (isStreaming) {
      // Connect to Next.js API socket server (which proxies gRPC backend)
      const socket = io({ path: "/api/v1/live/ws" });
      socketRef.current = socket;

      socket.emit("start_stream", { url: streamUrl });

      socket.on("frame_score", (data: FrameScore) => {
        setFrames((prev) => {
          const next = [...prev, data];
          if (next.length > 100) next.shift(); // Keep rolling last 100 frames
          return next;
        });

        if (data.confidence > 0.71) {
          setConsecutiveViolations((c) => {
            const nextCount = c + 1;
            if (nextCount >= 10) {
              setLastViolationFrame(data.frame_number);
            }
            return nextCount;
          });
        } else {
          setConsecutiveViolations(0);
        }
      });

      return () => {
        socket.disconnect();
      };
    } else {
      if (socketRef.current) {
        socketRef.current.emit("stop_stream");
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    }
  }, [isStreaming, streamUrl]);

  // Simulating live frames when running locally without a live stream source
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isStreaming && !socketRef.current) {
      let frameNum = 1;
      interval = setInterval(() => {
        const anomalyTrigger = frameNum > 35 && frameNum < 65;
        const confidence = anomalyTrigger 
          ? parseFloat(Math.min(1.0, Math.random() * 0.2 + 0.8).toFixed(2))
          : parseFloat((Math.random() * 0.15 + 0.1).toFixed(2));
          
        const scoreData: FrameScore = {
          frame_number: frameNum,
          confidence,
          anomaly_flags: confidence > 0.71 ? ["FACIAL_WARP", "TEMPORAL_JITTER"] : [],
          timestamp: new Date().toLocaleTimeString()
        };
        
        setFrames((prev) => {
          const next = [...prev, scoreData];
          if (next.length > 100) next.shift();
          return next;
        });

        if (confidence > 0.71) {
          setConsecutiveViolations((c) => {
            const nextCount = c + 1;
            if (nextCount >= 10) {
              setLastViolationFrame(frameNum);
            }
            return nextCount;
          });
        } else {
          setConsecutiveViolations(0);
        }

        frameNum++;
      }, 500);
    }
    return () => clearInterval(interval);
  }, [isStreaming]);

  const handleStart = () => {
    setFrames([]);
    setConsecutiveViolations(0);
    setLastViolationFrame(null);
    setIsStreaming(true);
  };

  const handleStop = () => {
    setIsStreaming(false);
  };

  const isAlarmTriggered = consecutiveViolations >= 10;

  return (
    <div className="space-y-6">
      
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white font-mono">LIVE CALL DEEPFAKE MONITOR</h1>
          <p className="text-xs text-gray-400">Bidirectional gRPC streaming analysis of active VoIP / Video conference streams</p>
        </div>
        <div className="flex items-center gap-2">
          {isStreaming ? (
            <span className="flex items-center gap-1.5 px-3 py-1 rounded bg-red-500/10 border border-red-500/20 text-red-500 text-xs font-mono font-bold animate-pulse">
              <RefreshCw size={12} className="animate-spin" /> STREAM ACTIVE
            </span>
          ) : (
            <span className="px-3 py-1 rounded bg-gray-500/10 border border-gray-500/20 text-gray-500 text-xs font-mono font-bold">
              STANDBY
            </span>
          )}
        </div>
      </div>

      {/* Alarm Banner */}
      {isAlarmTriggered && (
        <div className="bg-red-950 border-2 border-red-500 text-red-400 p-4 rounded-xl flex items-center gap-3 animate-bounce shadow-lg shadow-red-500/10">
          <AlertTriangle size={24} className="text-red-500 animate-pulse" />
          <div className="font-mono text-xs">
            <span className="font-extrabold text-sm block">HIGH CONFIDENCE DEEPFAKE DETECTED</span>
            Session exhibits continuous synthesis anomalies (Frame: {lastViolationFrame})
          </div>
        </div>
      )}

      {/* Setup Config */}
      <div className="bg-surface border border-border p-5 rounded-xl flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2 flex-1 min-w-[300px]">
          <Settings size={16} className="text-gray-500" />
          <span className="text-xs font-mono text-gray-400 uppercase mr-2">SOURCE URL:</span>
          <input 
            type="text"
            value={streamUrl}
            onChange={(e) => setStreamUrl(e.target.value)}
            disabled={isStreaming}
            className="bg-background border border-border rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-forensic-blue flex-1"
          />
        </div>
        <div className="flex gap-3">
          {!isStreaming ? (
            <button 
              onClick={handleStart}
              className="flex items-center gap-2 bg-forensic-blue hover:bg-blue-600 px-4 py-2 rounded-lg text-xs font-mono font-bold text-white transition-all shadow shadow-blue-500/10"
            >
              <Play size={14} /> Start Monitor
            </button>
          ) : (
            <button 
              onClick={handleStop}
              className="flex items-center gap-2 bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg text-xs font-mono font-bold text-white transition-all shadow shadow-red-500/10"
            >
              <Square size={14} /> Kill Connection
            </button>
          )}
        </div>
      </div>

      {/* Rolling Charts */}
      <div className="bg-surface border border-border p-5 rounded-xl">
        <h3 className="text-xs font-semibold uppercase tracking-wider mb-4 font-mono text-gray-300">
          Rolling Synthesis Likelihood Timeline (gRPC stream confidence metrics)
        </h3>
        <div className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={frames}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E1E2E" />
              <XAxis dataKey="frame_number" stroke="#94A3B8" fontSize={9} />
              <YAxis stroke="#94A3B8" domain={[0, 1.0]} fontSize={9} />
              <Tooltip contentStyle={{ backgroundColor: "#12121A", borderColor: "#1E1E2E" }} />
              <Line type="monotone" dataKey="confidence" stroke="#EF4444" strokeWidth={2.5} dot={false} name="Deepfake Score" />
              <Line type="monotone" dataKey="confidence" stroke="#EF4444" strokeDasharray="5 5" dot={false} strokeWidth={1} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Frame Table */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <h3 className="text-xs font-semibold uppercase tracking-wider mb-4 font-mono text-gray-300">
          Streaming Frame Evaluation Log
        </h3>
        <div className="overflow-y-auto max-h-60">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-gray-500">
                <th className="py-2">FRAME #</th>
                <th className="py-2">CONFIDENCE</th>
                <th className="py-2">ANOMALY SIGNATURES</th>
                <th className="py-2">TIMESTAMP</th>
              </tr>
            </thead>
            <tbody>
              {frames.slice().reverse().map((f) => (
                <tr key={f.frame_number} className="border-b border-border/40 hover:bg-white/5">
                  <td className="py-2 text-forensic-blue font-bold">#{f.frame_number}</td>
                  <td className={`py-2 font-bold ${f.confidence > 0.71 ? "text-forensic-danger" : "text-forensic-green"}`}>
                    {(f.confidence * 100).toFixed(0)}%
                  </td>
                  <td className="py-2 text-gray-300">
                    {f.anomaly_flags.length > 0 ? (
                      <span className="text-forensic-danger font-extrabold">{f.anomaly_flags.join(" | ")}</span>
                    ) : (
                      <span className="text-gray-500">NOMINAL</span>
                    )}
                  </td>
                  <td className="py-2 text-gray-500">{f.timestamp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}

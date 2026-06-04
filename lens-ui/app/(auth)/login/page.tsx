// app/(auth)/login/page.tsx
"use client";
import React from "react";
import { signIn } from "next-auth/react";
import { ShieldAlert } from "lucide-react";

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6 text-gray-200">
      <div className="w-full max-w-md bg-surface border border-border p-8 rounded-xl shadow-2xl relative overflow-hidden">
        
        {/* Neon blue top accent glow */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-forensic-blue to-cyan-400" />
        
        <div className="text-center mb-8">
          <span className="inline-block text-[10px] uppercase tracking-widest bg-red-600/10 border border-red-600/30 text-red-500 px-3 py-1 rounded font-mono font-bold mb-4">
            CONFIDENTIAL // FOR OFFICIAL USE ONLY
          </span>
          <div className="flex items-center justify-center gap-2 mb-2">
            <ShieldAlert size={28} className="text-forensic-blue" />
            <h1 className="text-3xl font-extrabold tracking-wider font-mono text-white">LENS</h1>
          </div>
          <p className="text-xs text-gray-400 font-sans tracking-wide uppercase">
            Layered Evidence &amp; Neural Synthesis Tracker
          </p>
        </div>

        <div className="space-y-4">
          <p className="text-sm text-center text-gray-400 font-sans">
            Federal Deepfake Analysis and Forensic Synthesis Ledger
          </p>
          
          <button 
            onClick={() => signIn("keycloak", { callbackUrl: "/" })}
            className="w-full bg-forensic-blue hover:bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-all shadow-md shadow-blue-500/10"
          >
            Sign in with Keycloak
          </button>
        </div>

        <div className="mt-8 pt-6 border-t border-border flex items-center justify-between text-[10px] text-gray-500 font-mono">
          <span>SECURE SYSTEM TLS 1.3</span>
          <span>v1.4.2</span>
        </div>
      </div>
    </div>
  );
}

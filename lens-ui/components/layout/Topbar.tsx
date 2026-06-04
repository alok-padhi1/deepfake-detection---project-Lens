// components/layout/Topbar.tsx
"use client";
import React from "react";
import { LogOut, User } from "lucide-react";
import { signOut } from "next-auth/react";

interface TopbarProps {
  user: {
    name?: string | null;
    email?: string | null;
    roles?: string[];
  };
}

export default function Topbar({ user }: TopbarProps) {
  const primaryRole = user.roles && user.roles.length > 0 ? user.roles[0] : "Submitter";

  return (
    <header className="h-16 border-b border-border bg-surface flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <span className="text-xs uppercase tracking-widest bg-red-600/10 border border-red-600/30 text-red-500 px-2.5 py-0.5 rounded font-mono font-bold">
          CONFIDENTIAL // NOFORN
        </span>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <div className="bg-white/5 border border-border p-1.5 rounded-full">
            <User size={16} className="text-gray-300" />
          </div>
          <div className="text-left">
            <div className="text-xs font-semibold text-white">{user.name || user.email}</div>
            <div className="text-[10px] text-forensic-blue font-mono uppercase">{primaryRole}</div>
          </div>
        </div>

        <button 
          onClick={() => signOut()} 
          className="text-gray-400 hover:text-white transition-colors"
          title="Sign Out"
        >
          <LogOut size={18} />
        </button>
      </div>
    </header>
  );
}

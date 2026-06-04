// components/layout/Sidebar.tsx
"use client";
import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, FolderOpen, Upload, Video, ShieldAlert, 
  Settings, Users, Activity, ChevronLeft, ChevronRight 
} from "lucide-react";

interface SidebarProps {
  roles: string[];
}

export default function Sidebar({ roles }: SidebarProps) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const isAdmin = roles.includes("Admin");

  const navItems = [
    { name: "Overview", path: "/", icon: LayoutDashboard },
    { name: "Submit Media", path: "/submit", icon: Upload },
    { name: "Cases", path: "/cases", icon: FolderOpen },
    { name: "Live Monitor", path: "/live", icon: Video },
  ];

  const adminItems = [
    { name: "Admin Overview", path: "/admin", icon: ShieldAlert },
    { name: "Users", path: "/admin/users", icon: Users },
    { name: "System Settings", path: "/admin/system", icon: Settings },
    { name: "Audit Ledger", path: "/admin/audit", icon: Activity },
  ];

  return (
    <aside className={`bg-surface border-r border-border min-h-screen transition-all duration-300 flex flex-col justify-between ${collapsed ? "w-16" : "w-60"}`}>
      <div>
        <div className="h-16 flex items-center justify-between px-4 border-b border-border">
          {!collapsed && <span className="font-bold text-forensic-blue tracking-wider font-mono">LENS PLATFORM</span>}
          <button onClick={() => setCollapsed(!collapsed)} className="text-gray-400 hover:text-white">
            {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>
        
        <nav className="p-2 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.path;
            return (
              <Link key={item.path} href={item.path} className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${active ? "bg-forensic-blue/15 text-forensic-blue font-semibold border-l-2 border-forensic-blue" : "text-gray-400 hover:bg-white/5 hover:text-white"}`}>
                <Icon size={18} />
                {!collapsed && <span>{item.name}</span>}
              </Link>
            );
          })}
          
          {isAdmin && (
            <>
              <div className="pt-4 pb-1 border-t border-border mt-4">
                {!collapsed && <span className="text-[10px] text-gray-500 uppercase tracking-widest px-3">Administration</span>}
              </div>
              {adminItems.map((item) => {
                const Icon = item.icon;
                const active = pathname === item.path;
                return (
                  <Link key={item.path} href={item.path} className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${active ? "bg-forensic-amber/15 text-forensic-amber font-semibold border-l-2 border-forensic-amber" : "text-gray-400 hover:bg-white/5 hover:text-white"}`}>
                    <Icon size={18} />
                    {!collapsed && <span>{item.name}</span>}
                  </Link>
                );
              })}
            </>
          )}
        </nav>
      </div>

      <div className="p-4 border-t border-border text-center text-xs text-gray-500">
        {!collapsed && <span>V1.4.2 [CONFIDENTIAL]</span>}
      </div>
    </aside>
  );
}

// app/(dashboard)/admin/users/page.tsx
"use client";
import React, { useState } from "react";
import { Users, UserPlus, Trash, ShieldCheck } from "lucide-react";

export default function AdminUsersPage() {
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("Submitter");

  const [users, setUsers] = useState([
    { sub: "usr-01", username: "analyst_jones", email: "jones@forensics.gov", role: "Analyst", last_login: "2026-05-17 15:02", status: "ACTIVE" },
    { sub: "usr-02", username: "admin_smith", email: "smith@forensics.gov", role: "Admin", last_login: "2026-05-17 15:45", status: "ACTIVE" },
    { sub: "usr-03", username: "submitter_alpha", email: "alpha@agencies.gov", role: "Submitter", last_login: "2026-05-16 11:20", status: "ACTIVE" },
    { sub: "usr-04", username: "analyst_doe", email: "doe@forensics.gov", role: "Analyst", last_login: "2026-05-15 09:12", status: "SUSPENDED" }
  ]);

  const handleRoleChange = (sub: string, newRole: string) => {
    setUsers(prev => prev.map(u => u.sub === sub ? { ...u, role: newRole } : u));
  };

  const handleRevoke = (sub: string) => {
    setUsers(prev => prev.map(u => u.sub === sub ? { ...u, status: "SUSPENDED" } : u));
  };

  const handleInviteSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inviteEmail) return;
    const newUser = {
      sub: `usr-${Date.now()}`,
      username: inviteEmail.split("@")[0],
      email: inviteEmail,
      role: inviteRole,
      last_login: "NEVER",
      status: "ACTIVE"
    };
    setUsers(prev => [...prev, newUser]);
    setInviteEmail("");
  };

  return (
    <div className="space-y-6">
      
      {/* Invite form */}
      <form onSubmit={handleInviteSubmit} className="bg-surface border border-border p-5 rounded-xl space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider font-mono text-gray-300 flex items-center gap-1.5">
          <UserPlus size={16} className="text-forensic-blue" /> Send New OIDC Realm Invite
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-end">
          <div>
            <label className="block text-[10px] uppercase text-gray-500 tracking-wider mb-2">Target Email</label>
            <input 
              type="email"
              value={inviteEmail}
              onChange={(e) => setInviteEmail(e.target.value)}
              placeholder="e.g. analyst@agency.gov"
              className="w-full bg-background border border-border rounded-lg p-2 text-xs text-white focus:outline-none focus:border-forensic-blue"
            />
          </div>
          <div>
            <label className="block text-[10px] uppercase text-gray-500 tracking-wider mb-2">Default RBAC Role</label>
            <select 
              value={inviteRole}
              onChange={(e) => setInviteRole(e.target.value)}
              className="w-full bg-background border border-border rounded-lg p-2 text-xs text-white focus:outline-none focus:border-forensic-blue"
            >
              <option value="Submitter">Submitter</option>
              <option value="Analyst">Analyst</option>
              <option value="Admin">Admin</option>
            </select>
          </div>
          <button 
            type="submit"
            className="bg-forensic-blue hover:bg-blue-600 text-white font-mono font-bold text-xs py-2 px-4 rounded-lg h-9 transition-colors"
          >
            Dispatch Invitation
          </button>
        </div>
      </form>

      {/* Users table */}
      <div className="bg-surface border border-border rounded-xl p-5 space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider font-mono text-gray-300">
          👥 Registered Keycloak Directory Subscriptions
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-gray-500 uppercase tracking-wider">
                <th className="py-2.5">USERNAME</th>
                <th className="py-2.5">EMAIL</th>
                <th className="py-2.5">ROLE MAP</th>
                <th className="py-2.5">STATUS</th>
                <th className="py-2.5">LAST LOGIN</th>
                <th className="py-2.5 text-center">ACTIONS</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.sub} className="border-b border-border/40 hover:bg-white/5">
                  <td className="py-3 text-white font-bold">{u.username}</td>
                  <td className="py-3 text-gray-400">{u.email}</td>
                  <td className="py-3">
                    <select 
                      value={u.role}
                      onChange={(e) => handleRoleChange(u.sub, e.target.value)}
                      className="bg-background border border-border rounded px-2 py-0.5 text-xs text-forensic-blue focus:outline-none"
                    >
                      <option value="Submitter">Submitter</option>
                      <option value="Analyst">Analyst</option>
                      <option value="Admin">Admin</option>
                    </select>
                  </td>
                  <td className="py-3">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${u.status === "ACTIVE" ? "bg-green-500/10 text-green-500" : "bg-red-500/10 text-red-500"}`}>
                      {u.status}
                    </span>
                  </td>
                  <td className="py-3 text-gray-500">{u.last_login}</td>
                  <td className="py-3 text-center">
                    <button 
                      onClick={() => handleRevoke(u.sub)}
                      disabled={u.status === "SUSPENDED"}
                      className="inline-flex items-center gap-1 bg-red-500/10 hover:bg-red-500/20 disabled:bg-gray-700/10 disabled:text-gray-600 text-red-500 px-2 py-1 rounded text-[10px] font-bold border border-red-500/20 transition-all"
                    >
                      <Trash size={10} /> Suspend
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

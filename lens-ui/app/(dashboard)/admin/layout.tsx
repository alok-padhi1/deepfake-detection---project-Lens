// app/(dashboard)/admin/layout.tsx
import React from "react";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";

export default async function AdminLayout({ children }: { children: React.ReactNode }) {
  const session = await auth();
  
  if (!session) {
    redirect("/login");
  }

  const roles: string[] = (session as any).user?.roles || [];
  if (!roles.includes("Admin")) {
    // Non-Admin users are kicked out to general dashboard overview immediately
    redirect("/");
  }

  return (
    <div className="space-y-6">
      <div className="border-b border-border pb-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold font-mono tracking-wider text-forensic-amber uppercase">
            🛡️ Administrative Control Matrix
          </h1>
          <p className="text-xs text-gray-500">Configure global classification thresholds, trigger model retrainings, and review system ledgers</p>
        </div>
      </div>
      {children}
    </div>
  );
}

// app/(dashboard)/layout.tsx
import React from "react";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import Sidebar from "@/components/layout/Sidebar";
import Topbar from "@/components/layout/Topbar";

export default async function DashboardLayout({ children }: { children: React.ReactNode }) {
  const session = await auth();
  if (!session) {
    redirect("/login");
  }

  const user = {
    name: session.user?.name,
    email: session.user?.email,
    roles: (session as any).user?.roles || [],
  };

  return (
    <div className="flex bg-background min-h-screen text-gray-200">
      <Sidebar roles={user.roles} />
      <div className="flex-1 flex flex-col min-h-screen">
        <Topbar user={user} />
        <main className="flex-1 p-6 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
}

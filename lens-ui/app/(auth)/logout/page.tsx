// app/(auth)/logout/page.tsx
"use client";
import { useEffect } from "react";
import { signOut } from "next-auth/react";

export default function LogoutPage() {
  useEffect(() => {
    signOut({ callbackUrl: "/login" });
  }, []);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center text-gray-400 font-mono text-sm">
      TERMINATING SESSION AND RE-ENCRYPTING DATA LEDGER...
    </div>
  );
}

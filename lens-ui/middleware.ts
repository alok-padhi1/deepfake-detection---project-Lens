// middleware.ts
import { NextResponse } from "next/server";
import { auth } from "./lib/auth";

export default auth((req) => {
  const isLoggedIn = !!req.auth;
  const isAuthPage = req.nextUrl.pathname.startsWith("/login");
  const isAdminPage = req.nextUrl.pathname.startsWith("/admin");
  
  if (isAuthPage) {
    if (isLoggedIn) {
      return NextResponse.redirect(new URL("/", req.nextUrl));
    }
    return NextResponse.next();
  }

  if (!isLoggedIn) {
    let from = req.nextUrl.pathname;
    if (req.nextUrl.search) {
      from += req.nextUrl.search;
    }
    return NextResponse.redirect(
      new URL(`/login?from=${encodeURIComponent(from)}`, req.nextUrl)
    );
  }

  // Guard admin panel from non-Admin users
  if (isAdminPage) {
    const roles: string[] = (req.auth?.user as any)?.roles || [];
    if (!roles.includes("Admin")) {
      return NextResponse.redirect(new URL("/", req.nextUrl));
    }
  }

  return NextResponse.next();
});

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};

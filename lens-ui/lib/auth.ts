// lib/auth.ts
import NextAuth from "next-auth";
import KeycloakProvider from "next-auth/providers/keycloak";

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    KeycloakProvider({
      clientId: process.env.KEYCLOAK_CLIENT_ID || "lens-ui",
      clientSecret: process.env.KEYCLOAK_CLIENT_SECRET || "",
      issuer: `${process.env.KEYCLOAK_URL}/realms/${process.env.KEYCLOAK_REALM}`,
      authorization: {
        params: { scope: "openid email profile" },
        url: `${process.env.NEXT_PUBLIC_KEYCLOAK_URL_EXTERNAL || 'http://localhost:8080'}/realms/${process.env.KEYCLOAK_REALM || 'lens'}/protocol/openid-connect/auth`
      },
      token: `${process.env.KEYCLOAK_URL || 'http://keycloak:8080'}/realms/${process.env.KEYCLOAK_REALM || 'lens'}/protocol/openid-connect/token`,
      userinfo: `${process.env.KEYCLOAK_URL || 'http://keycloak:8080'}/realms/${process.env.KEYCLOAK_REALM || 'lens'}/protocol/openid-connect/userinfo`,
      jwks_endpoint: `${process.env.KEYCLOAK_URL || 'http://keycloak:8080'}/realms/${process.env.KEYCLOAK_REALM || 'lens'}/protocol/openid-connect/certs`
    }),
  ],
  callbacks: {
    async jwt({ token, account, profile }: any) {
      if (account && profile) {
        token.accessToken = account.access_token;
        token.idToken = account.id_token;
        // Extract roles from realm access scope inside token
        const realmAccess = profile.realm_access || {};
        token.roles = realmAccess.roles || [];
      }
      return token;
    },
    async session({ session, token }: any) {
      session.accessToken = token.accessToken;
      session.user.roles = token.roles || [];
      return session;
    },
  },
  pages: {
    signIn: "/login",
    signOut: "/logout",
  },
});

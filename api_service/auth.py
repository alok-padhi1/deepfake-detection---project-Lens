# auth.py
import json
import logging
import requests
from typing import Dict, List, Optional
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from config import settings

log = logging.getLogger("auth")

security_bearer = HTTPBearer()

class UserPrincipal:
    def __init__(self, sub: str, username: str, roles: List[str], email: str):
        self.sub = sub
        self.username = username
        self.roles = roles
        self.email = email

    def has_role(self, role: str) -> bool:
        return role in self.roles

class AuthManager:
    def __init__(self):
        self._jwks: Dict = {}
        self._load_jwks()

    def _load_jwks(self):
        try:
            resp = requests.get(settings.KEYCLOAK_JWKS_URL, timeout=5.0)
            if resp.status_code == 200:
                self._jwks = resp.json()
                log.info("Successfully fetched OIDC JWKS keys from Keycloak.")
        except Exception as e:
            log.warning(f"Failed to fetch JWKS from Keycloak: {e}. Key verification might fail offline.")

    def get_public_key(self, token: str) -> Optional[Dict]:
        try:
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            if not self._jwks:
                self._load_jwks()
            for key in self._jwks.get("keys", []):
                if key.get("kid") == kid:
                    return key
        except Exception as e:
            log.error(f"Error parsing kid from token: {e}")
        return None

    def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security_bearer)) -> UserPrincipal:
        token = credentials.credentials
        pub_key = self.get_public_key(token)
        if not pub_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token header key ID (kid) or Keycloak unreachable."
            )
        try:
            # Validate signature, expiry, issuer and audience
            payload = jwt.decode(
                token,
                pub_key,
                algorithms=["RS256"],
                audience=settings.KEYCLOAK_CLIENT_ID,
                issuer=f"{settings.KEYCLOAK_URL}/realms/{settings.KEYCLOAK_REALM}"
            )
            
            sub = payload.get("sub")
            username = payload.get("preferred_username") or payload.get("sub")
            email = payload.get("email", "")
            
            # Keycloak roles mapping
            realm_access = payload.get("realm_access", {})
            roles = realm_access.get("roles", [])
            
            return UserPrincipal(sub=sub, username=username, roles=roles, email=email)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired.")
        except jwt.JWTClaimsError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Claim validation failed: {e}")
        except JWTError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token verification failed: {e}")

auth_manager = AuthManager()

# RBAC Route Dependency Helpers
def require_role(allowed_roles: List[str]):
    def dependency(user: UserPrincipal = Depends(auth_manager.verify_token)):
        if not any(user.has_role(r) for r in allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {allowed_roles}"
            )
        return user
    return dependency

def get_current_user(user: UserPrincipal = Depends(auth_manager.verify_token)) -> UserPrincipal:
    return user

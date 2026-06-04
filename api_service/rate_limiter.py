# rate_limiter.py
import time
import logging
import redis
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config import settings

log = logging.getLogger("rate_limiter")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class TokenBucketRateLimiter:
    def __init__(self):
        self._redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD
        )
        self.capacity = 100.0
        self.refill_rate = 100.0 / 60.0  # 1.666 tokens per second

    def is_allowed(self, api_key: str) -> bool:
        if not api_key:
            return True # If no API key, bypass to auth handler directly
        
        redis_key = f"rate_limit:analyze:{api_key}"
        now = time.time()
        
        try:
            # Multi/exec transaction pipeline to ensure atomic token updates
            pipe = self._redis.pipeline()
            pipe.hmget(redis_key, "tokens", "last_updated")
            res = pipe.execute()[0]
            
            tokens_val, last_updated_val = res[0], res[1]
            
            if tokens_val is None or last_updated_val is None:
                # First time seeing this API key
                tokens = self.capacity
                last_updated = now
            else:
                tokens = float(tokens_val)
                last_updated = float(last_updated_val)
                
            # Refill tokens
            elapsed = now - last_updated
            refilled = tokens + (elapsed * self.refill_rate)
            tokens = min(self.capacity, refilled)
            last_updated = now
            
            if tokens >= 1.0:
                tokens -= 1.0
                # Save updated bucket
                pipe = self._redis.pipeline()
                pipe.hset(redis_key, mapping={"tokens": str(tokens), "last_updated": str(last_updated)})
                pipe.expire(redis_key, 60)
                pipe.execute()
                return True
            else:
                return False
        except Exception as e:
            log.error(f"Redis rate limiter exception: {e}. Falling back to default permit.")
            return True

rate_limiter = TokenBucketRateLimiter()

def require_rate_limit(api_key: str = Security(api_key_header)):
    if api_key and not rate_limiter.is_allowed(api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 100 requests per minute."
        )

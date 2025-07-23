"""
Rate limiting utility for OpenAI API calls.
Provides configurable rate limiting with exponential backoff and request tracking.
"""

import time
import logging
from typing import Callable, Any, Optional, Dict
from functools import wraps
import threading
from collections import deque
from datetime import datetime, timedelta
import openai
from openai import RateLimitError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter for API calls with configurable limits."""
    
    def __init__(
        self,
        requests_per_minute: int = 50,
        tokens_per_minute: int = 150000,
        min_delay_between_requests: float = 0.1,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
            tokens_per_minute: Maximum number of tokens per minute (estimated)
            min_delay_between_requests: Minimum delay between consecutive requests in seconds
            max_retries: Maximum number of retries for rate-limited requests
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.min_delay_between_requests = min_delay_between_requests
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Request tracking
        self._request_times = deque(maxlen=requests_per_minute)
        self._last_request_time = 0
        
        # Token tracking (approximate)
        self._token_usage = deque(maxlen=100)  # Track last 100 requests
        self._total_tokens_used = 0
        
        logger.info(f"Initialized RateLimiter: {requests_per_minute} req/min, {tokens_per_minute} tokens/min")
    
    def _wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        with self._lock:
            current_time = time.time()
            
            # Enforce minimum delay between requests
            time_since_last = current_time - self._last_request_time
            if time_since_last < self.min_delay_between_requests:
                sleep_time = self.min_delay_between_requests - time_since_last
                logger.debug(f"Sleeping for {sleep_time:.2f}s to respect min delay")
                time.sleep(sleep_time)
                current_time = time.time()
            
            # Check requests per minute limit
            minute_ago = current_time - 60
            recent_requests = [t for t in self._request_times if t > minute_ago]
            
            if len(recent_requests) >= self.requests_per_minute:
                # Calculate how long to wait
                oldest_request = min(recent_requests)
                wait_time = 60 - (current_time - oldest_request) + 0.1
                logger.info(f"Rate limit approaching: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                current_time = time.time()
            
            # Update tracking
            self._request_times.append(current_time)
            self._last_request_time = current_time
    
    def _handle_rate_limit_error(self, error: RateLimitError, attempt: int) -> float:
        """Calculate backoff time for rate limit errors."""
        # Try to extract retry-after from error
        retry_after = getattr(error, 'retry_after', None)
        
        if retry_after:
            backoff_time = float(retry_after)
            logger.warning(f"Rate limited. Retry after: {backoff_time}s")
        else:
            # Exponential backoff
            backoff_time = min(
                self.initial_backoff * (2 ** attempt),
                self.max_backoff
            )
            logger.warning(f"Rate limited. Backing off for {backoff_time}s (attempt {attempt})")
        
        return backoff_time
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply rate limiting to a function."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(self.max_retries):
                try:
                    # Wait if necessary before making request
                    self._wait_if_needed()
                    
                    # Make the API call
                    result = func(*args, **kwargs)
                    
                    # Track approximate token usage if available
                    if hasattr(result, 'usage'):
                        with self._lock:
                            tokens = result.usage.total_tokens
                            self._token_usage.append((time.time(), tokens))
                            self._total_tokens_used += tokens
                    
                    return result
                    
                except RateLimitError as e:
                    if attempt >= self.max_retries - 1:
                        logger.error(f"Max retries ({self.max_retries}) exceeded for rate-limited request")
                        raise
                    
                    backoff_time = self._handle_rate_limit_error(e, attempt)
                    time.sleep(backoff_time)
                    
                except Exception as e:
                    # Re-raise non-rate-limit errors
                    logger.error(f"Non-rate-limit error in API call: {e}")
                    raise
            
            # Should not reach here
            raise Exception("Rate limiter logic error")
        
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics."""
        with self._lock:
            current_time = time.time()
            minute_ago = current_time - 60
            recent_requests = [t for t in self._request_times if t > minute_ago]
            
            # Calculate token rate
            recent_tokens = sum(
                tokens for timestamp, tokens in self._token_usage 
                if timestamp > minute_ago
            )
            
            return {
                "requests_in_last_minute": len(recent_requests),
                "tokens_in_last_minute": recent_tokens,
                "total_tokens_used": self._total_tokens_used,
                "requests_remaining": max(0, self.requests_per_minute - len(recent_requests)),
                "current_rpm_limit": self.requests_per_minute,
                "current_tpm_limit": self.tokens_per_minute
            }


# Global rate limiter instance with conservative defaults
default_rate_limiter = RateLimiter(
    requests_per_minute=50,  # Conservative default
    tokens_per_minute=150000,
    min_delay_between_requests=0.2  # 200ms between requests
)


# Convenience decorators for common API calls
rate_limited_chat_completion = default_rate_limiter
rate_limited_embedding = default_rate_limiter


def create_rate_limited_client(
    api_key: str,
    requests_per_minute: int = 50,
    tokens_per_minute: int = 150000
) -> 'RateLimitedOpenAIClient':
    """Create a rate-limited OpenAI client wrapper."""
    return RateLimitedOpenAIClient(
        api_key=api_key,
        rate_limiter=RateLimiter(
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute
        )
    )


class RateLimitedOpenAIClient:
    """Wrapper for OpenAI client with built-in rate limiting."""
    
    def __init__(self, api_key: str, rate_limiter: Optional[RateLimiter] = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.rate_limiter = rate_limiter or default_rate_limiter
        
        # Wrap methods
        self.chat = self._create_chat_wrapper()
        self.embeddings = self._create_embeddings_wrapper()
    
    def _create_chat_wrapper(self):
        """Create rate-limited chat completions wrapper."""
        class ChatWrapper:
            def __init__(self, client, rate_limiter):
                self._client = client
                self._rate_limiter = rate_limiter
            
            @property
            def completions(self):
                class CompletionsWrapper:
                    def __init__(self, client, rate_limiter):
                        self._client = client
                        self._rate_limiter = rate_limiter
                    
                    def create(self, **kwargs):
                        @self._rate_limiter
                        def _create():
                            return self._client.chat.completions.create(**kwargs)
                        return _create()
                
                return CompletionsWrapper(self._client, self._rate_limiter)
        
        return ChatWrapper(self.client, self.rate_limiter)
    
    def _create_embeddings_wrapper(self):
        """Create rate-limited embeddings wrapper."""
        class EmbeddingsWrapper:
            def __init__(self, client, rate_limiter):
                self._client = client
                self._rate_limiter = rate_limiter
            
            def create(self, **kwargs):
                @self._rate_limiter
                def _create():
                    return self._client.embeddings.create(**kwargs)
                return _create()
        
        return EmbeddingsWrapper(self.client, self.rate_limiter)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return self.rate_limiter.get_stats()
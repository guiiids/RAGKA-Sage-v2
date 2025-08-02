"""
Redis Service for RAGKA

This module provides a simple interface to Redis operations for caching in the RAGKA system.
"""

import os
import json
import logging
import redis
from typing import Any, Dict, Optional, List, Union

# Configure logging
logger = logging.getLogger()

class RedisService:
    def get_current_timestamp(self) -> int:
        """
        Return the current UTC timestamp as an integer.
        """
        import time
        return int(time.time())
    """
    Service for interacting with Redis cache.
    Provides methods for getting, setting, and deleting cache entries,
    as well as health checks and connection management.
    """
    
    def __init__(self):
        """Initialize the Redis service with connection parameters from environment variables."""
        # Get Redis connection parameters from environment variables
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.db = int(os.getenv("REDIS_DB", "0"))
        self.password = os.getenv("REDIS_PASSWORD", None)
        self.default_expiration = int(os.getenv("REDIS_DEFAULT_EXPIRATION", "3600"))  # 1 hour default
        
        # Initialize Redis client
        self._client = None
        self._connected = False
        
        # Try to connect to Redis
        try:
            self._connect()
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {str(e)}")
    
    def _connect(self) -> None:
        """
        Connect to Redis server.
        
        Raises:
            redis.RedisError: If connection fails
        """
        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=5,
                decode_responses=False  # We'll handle decoding ourselves
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except redis.RedisError as e:
            self._connected = False
            logger.error(f"Redis connection error: {str(e)}")
            raise
    
    def is_connected(self) -> bool:
        """
        Check if connected to Redis.
        
        Returns:
            True if connected, False otherwise
        """
        if not self._connected or not self._client:
            return False
        
        try:
            self._client.ping()
            return True
        except:
            self._connected = False
            return False
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to Redis if not connected.
        
        Returns:
            True if reconnected successfully, False otherwise
        """
        if self.is_connected():
            return True
        
        try:
            self._connect()
            return True
        except:
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from Redis.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found
        """
        if not self.is_connected() and not self.reconnect():
            return None
        
        try:
            value = self._client.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except:
                # If not JSON, return as is
                return value
        except Exception as e:
            logger.error(f"Error getting from Redis: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, expiration: Optional[int] = None) -> bool:
        """
        Set a value in Redis.
        
        Args:
            key: The cache key
            value: The value to cache
            expiration: Time in seconds until expiration (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected() and not self.reconnect():
            return False
        
        try:
            # Serialize value if it's not a string
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
            
            # Use default expiration if not specified
            if expiration is None:
                expiration = self.default_expiration
            
            # Set in Redis
            self._client.set(key, value, ex=expiration)
            return True
        except Exception as e:
            logger.error(f"Error setting in Redis: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from Redis.
        
        Args:
            key: The cache key
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected() and not self.reconnect():
            return False
        
        try:
            result = self._client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting from Redis: {str(e)}")
            return False
    
    def flush_all(self) -> bool:
        """
        Flush all keys from the current database.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected() and not self.reconnect():
            return False
        
        try:
            self._client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Error flushing Redis: {str(e)}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Get Redis health information.
        
        Returns:
            Dictionary with Redis health information
        """
        if not self.is_connected() and not self.reconnect():
            return {"connected": False}
        
        try:
            info = self._client.info()
            return {
                "connected": True,
                "mode": info.get("redis_mode", "standalone"),
                "version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "clients_connected": info.get("connected_clients", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis health: {str(e)}")
            return {"connected": False, "error": str(e)}
    
    def keys(self, pattern: str) -> List[str]:
        """
        Get keys matching a pattern.
        
        Args:
            pattern: The pattern to match
            
        Returns:
            List of matching keys
        """
        if not self.is_connected() and not self.reconnect():
            return []
        
        try:
            keys = self._client.keys(pattern)
            # Convert bytes to strings
            return [k.decode('utf-8') if isinstance(k, bytes) else k for k in keys]
        except Exception as e:
            logger.error(f"Error getting Redis keys: {str(e)}")
            return []
    
    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: The pattern to match
            
        Returns:
            Number of keys deleted
        """
        if not self.is_connected() and not self.reconnect():
            return 0
        
        try:
            keys = self.keys(pattern)
            if not keys:
                return 0
            
            # Delete keys in batches to avoid blocking Redis
            deleted = 0
            batch_size = 100
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i+batch_size]
                deleted += self._client.delete(*batch)
            
            return deleted
        except Exception as e:
            logger.error(f"Error deleting Redis keys by pattern: {str(e)}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get Redis statistics.
        
        Returns:
            Dictionary with Redis statistics
        """
        if not self.is_connected() and not self.reconnect():
            return {"connected": False}
        
        try:
            info = self._client.info()
            stats = {
                "connected": True,
                "total_keys": self._client.dbsize(),
                "memory_used": info.get("used_memory_human", "unknown"),
                "memory_peak": info.get("used_memory_peak_human", "unknown"),
                "uptime": info.get("uptime_in_days", 0),
                "hit_rate": 0.0
            }
            
            # Calculate hit rate if available
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            if hits + misses > 0:
                stats["hit_rate"] = hits / (hits + misses)
            
            return stats
        except Exception as e:
            logger.error(f"Error getting Redis stats: {str(e)}")
            return {"connected": False, "error": str(e)}


# Create a singleton instance
redis_service = RedisService()

"""
Redis Citation Service for RAGKA

This module provides message-scoped citation storage using Redis.
Simple, clean approach: each message has its own citation array.
"""

import os
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional
from services.redis_service import redis_service

# Configure logging
logger = logging.getLogger()

class RedisCitationService:
    """
    Service for managing message-scoped citations in Redis.
    
    Key pattern: citation:{session_id}:{message_id} → [source1, source2, source3]
    
    This eliminates complex ID mapping and provides clean per-message citation numbering.
    """
    
    def __init__(self):
        """Initialize the Redis citation service."""
        self.citation_expiration = 43200  # 12 hours, same as RAG responses
        self.citation_prefix = "citation:"
        
        logger.info("Redis citation service initialized")
    
    def _generate_message_id(self, query: str, timestamp: float = None) -> str:
        """
        Generate a consistent message ID based on query and timestamp.
        
        Args:
            query: The user query
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            Consistent message ID
        """
        import time
        if timestamp is None:
            timestamp = time.time()
        
        # Create a hash from query and timestamp for consistency
        hash_input = f"{query}_{timestamp}"
        hash_obj = hashlib.md5(hash_input.encode('utf-8'))
        return f"msg_{hash_obj.hexdigest()[:8]}"
    
    def _generate_cache_key(self, session_id: str, message_id: str) -> str:
        """
        Generate cache key for message citations.
        
        Args:
            session_id: Session identifier
            message_id: Message identifier
            
        Returns:
            Redis cache key
        """
        return f"{self.citation_prefix}{session_id}:{message_id}"
    
    def store_message_citations(self, session_id: str, message_id: str, sources: List[Dict[str, Any]]) -> bool:
        """
        Store citations for a specific message.
        
        Args:
            session_id: Session identifier
            message_id: Message identifier
            sources: List of source dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not redis_service.is_connected():
            logger.warning("Redis not available for citation storage")
            return False
        
        try:
            cache_key = self._generate_cache_key(session_id, message_id)
            
            # Clean and prepare sources for storage
            clean_sources = []
            for i, source in enumerate(sources):
                if isinstance(source, dict):
                    clean_source = {
                        'index': i + 1,  # 1-based indexing for display
                        'title': source.get('title', f'Source {i + 1}'),
                        'content': source.get('content', ''),
                        'url': source.get('url', ''),
                        'id': source.get('id', f'source_{i + 1}'),
                        'display_id': str(i + 1)
                    }
                elif isinstance(source, str):
                    clean_source = {
                        'index': i + 1,
                        'title': f'Source {i + 1}',
                        'content': source,
                        'url': '',
                        'id': f'source_{i + 1}',
                        'display_id': str(i + 1)
                    }
                else:
                    continue
                
                clean_sources.append(clean_source)
            
            # Store in Redis
            success = redis_service.set(cache_key, clean_sources, self.citation_expiration)
            
            if success:
                logger.debug(f"Stored {len(clean_sources)} citations for message {message_id}")
            else:
                logger.error(f"Failed to store citations for message {message_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing citations: {str(e)}")
            return False
    
    def get_message_citations(self, session_id: str, message_id: str) -> List[Dict[str, Any]]:
        """
        Get citations for a specific message.
        
        Args:
            session_id: Session identifier
            message_id: Message identifier
            
        Returns:
            List of citation dictionaries
        """
        if not redis_service.is_connected():
            logger.warning("Redis not available for citation retrieval")
            return []
        
        try:
            cache_key = self._generate_cache_key(session_id, message_id)
            citations = redis_service.get(cache_key)
            
            if citations is None:
                logger.debug(f"No citations found for message {message_id}")
                return []
            
            if isinstance(citations, list):
                logger.debug(f"Retrieved {len(citations)} citations for message {message_id}")
                return citations
            else:
                logger.warning(f"Invalid citation data format for message {message_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving citations: {str(e)}")
            return []
    
    def clear_session_citations(self, session_id: str) -> bool:
        """
        Clear all citations for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not redis_service.is_connected():
            logger.warning("Redis not available for citation clearing")
            return False
        
        try:
            pattern = f"{self.citation_prefix}{session_id}:*"
            deleted_count = redis_service.delete_pattern(pattern)
            logger.info(f"Cleared {deleted_count} citation entries for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing session citations: {str(e)}")
            return False
    
    def clear_message_citations(self, session_id: str, message_id: str) -> bool:
        """
        Clear citations for a specific message.
        
        Args:
            session_id: Session identifier
            message_id: Message identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not redis_service.is_connected():
            logger.warning("Redis not available for citation clearing")
            return False
        
        try:
            cache_key = self._generate_cache_key(session_id, message_id)
            success = redis_service.delete(cache_key)
            
            if success:
                logger.debug(f"Cleared citations for message {message_id}")
            else:
                logger.debug(f"No citations found to clear for message {message_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error clearing message citations: {str(e)}")
            return False
    
    def get_citation_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get citation statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with citation statistics
        """
        if not redis_service.is_connected():
            return {"connected": False}
        
        try:
            pattern = f"{self.citation_prefix}{session_id}:*"
            keys = redis_service.keys(pattern)
            
            total_messages = len(keys)
            total_citations = 0
            
            for key in keys:
                citations = redis_service.get(key)
                if isinstance(citations, list):
                    total_citations += len(citations)
            
            return {
                "connected": True,
                "session_id": session_id,
                "total_messages_with_citations": total_messages,
                "total_citations": total_citations,
                "redis_keys": keys
            }
            
        except Exception as e:
            logger.error(f"Error getting citation stats: {str(e)}")
            return {"connected": False, "error": str(e)}


# Create a singleton instance
redis_citation_service = RedisCitationService()

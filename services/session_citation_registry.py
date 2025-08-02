"""
Session-Wide Citation Registry for RAGKA

This module provides session-wide persistent citation storage using Redis.
Each unique source gets a permanent citation ID that persists across all messages in a session.
"""

import os
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from services.redis_service import redis_service

# Configure logging
logger = logging.getLogger(__name__)

class SessionCitationRegistry:
    """
    Service for managing session-wide citations in Redis.
    
    Key patterns:
    - session:{session_id}:citations:registry → Hash of all unique sources with their citation IDs
    - session:{session_id}:citations:counter → Global citation counter for the session
    - session:{session_id}:citations:source:{citation_id} → Full source data
    - session:{session_id}:citations:lookup:{source_hash} → Maps source hash to citation ID
    
    This ensures consistent citation numbering across all messages in a session.
    """
    
    def __init__(self):
        """Initialize the session citation registry."""
        self.citation_expiration = 43200  # 12 hours
        self.registry_prefix = "session:"
        
        logger.info("Session citation registry initialized")
    
    def _generate_source_hash(self, source: Dict[str, Any]) -> str:
        """
        Generate a consistent hash for source deduplication.
        
        Args:
            source: Source dictionary containing title and content
            
        Returns:
            Hash string for the source
        """
        # Use title + first 500 chars of content for hash
        content = source.get('content', '')
        title = source.get('title', '')
        hash_input = f"{title}_{content[:500]}"
        
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()[:12]
    
    def _get_registry_key(self, session_id: str) -> str:
        """Get the registry key for a session."""
        return f"{self.registry_prefix}{session_id}:citations:registry"
    
    def _get_counter_key(self, session_id: str) -> str:
        """Get the counter key for a session."""
        return f"{self.registry_prefix}{session_id}:citations:counter"
    
    def _get_source_key(self, session_id: str, citation_id: int) -> str:
        """Get the source storage key for a citation ID."""
        return f"{self.registry_prefix}{session_id}:citations:source:{citation_id}"
    
    def _get_lookup_key(self, session_id: str, source_hash: str) -> str:
        """Get the lookup key for a source hash."""
        return f"{self.registry_prefix}{session_id}:citations:lookup:{source_hash}"
    
    def register_sources(self, session_id: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Register sources in the session citation registry and return sources with citation IDs.
        
        Args:
            session_id: Session identifier
            sources: List of source dictionaries
            
        Returns:
            List of sources with assigned citation IDs
        """
        if not redis_service.is_connected():
            logger.warning("Redis not available for citation registry")
            return self._fallback_sources(sources)
        
        try:
            registered_sources = []
            
            for source in sources:
                if not isinstance(source, dict):
                    continue
                
                # Generate hash for this source
                source_hash = self._generate_source_hash(source)
                lookup_key = self._get_lookup_key(session_id, source_hash)
                
                # Check if source already exists
                existing_citation_id = redis_service.get(lookup_key)
                
                if existing_citation_id:
                    # Source already exists, reuse citation ID
                    citation_id = int(existing_citation_id)
                    logger.debug(f"Reusing citation ID {citation_id} for existing source")
                else:
                    # New source, assign new citation ID
                    citation_id = self._get_next_citation_id(session_id)
                    
                    # Store the lookup mapping
                    redis_service.set(lookup_key, citation_id, self.citation_expiration)
                    
                    # Store the full source data
                    source_key = self._get_source_key(session_id, citation_id)
                    source_data = {
                        'citation_id': citation_id,
                        'title': source.get('title', f'Source {citation_id}'),
                        'content': source.get('content', ''),
                        'url': source.get('url', ''),
                        'id': source.get('id', f'source_{citation_id}'),
                        'hash': source_hash
                    }
                    redis_service.set(source_key, source_data, self.citation_expiration)
                    
                    logger.debug(f"Assigned new citation ID {citation_id} to source")
                
                # Add citation info to source
                registered_source = source.copy()
                registered_source.update({
                    'citation_id': citation_id,
                    'display_id': str(citation_id),
                    'session_id': session_id,
                    'hash': source_hash
                })
                
                registered_sources.append(registered_source)
            
            # Update registry summary
            self._update_registry_summary(session_id, registered_sources)
            
            logger.info(f"Registered {len(registered_sources)} sources for session {session_id}")
            return registered_sources
            
        except Exception as e:
            logger.error(f"Error registering sources: {str(e)}")
            return self._fallback_sources(sources)
    
    def _get_next_citation_id(self, session_id: str) -> int:
        """
        Get the next citation ID for the session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Next citation ID
        """
        counter_key = self._get_counter_key(session_id)
        
        try:
            # Increment counter atomically
            citation_id = redis_service.incr(counter_key)
            redis_service.expire(counter_key, self.citation_expiration)
            return citation_id
        except Exception as e:
            logger.error(f"Error getting next citation ID: {str(e)}")
            return 1
    
    def _update_registry_summary(self, session_id: str, sources: List[Dict[str, Any]]) -> None:
        """
        Update the registry summary with current sources.
        
        Args:
            session_id: Session identifier
            sources: List of registered sources
        """
        try:
            registry_key = self._get_registry_key(session_id)
            
            # Get existing registry or create new one
            existing_registry = redis_service.get(registry_key) or {}
            
            # Add new sources to registry
            for source in sources:
                citation_id = source.get('citation_id')
                if citation_id:
                    existing_registry[str(citation_id)] = {
                        'title': source.get('title', ''),
                        'hash': source.get('hash', ''),
                        'registered_at': redis_service.get_current_timestamp()
                    }
            
            # Store updated registry
            redis_service.set(registry_key, existing_registry, self.citation_expiration)
            
        except Exception as e:
            logger.error(f"Error updating registry summary: {str(e)}")
    
    def get_source_by_citation_id(self, session_id: str, citation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get source data by citation ID.
        
        Args:
            session_id: Session identifier
            citation_id: Citation ID to look up
            
        Returns:
            Source data if found, None otherwise
        """
        if not redis_service.is_connected():
            logger.warning("Redis not available for source lookup")
            return None
        
        try:
            source_key = self._get_source_key(session_id, citation_id)
            source_data = redis_service.get(source_key)
            
            if source_data:
                logger.debug(f"Found source for citation ID {citation_id}")
                return source_data
            else:
                logger.debug(f"No source found for citation ID {citation_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving source by citation ID: {str(e)}")
            return None
    
    def get_all_session_citations(self, session_id: str) -> Dict[str, Any]:
        """
        Get all citations for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with all session citations
        """
        if not redis_service.is_connected():
            return {"connected": False}
        
        try:
            registry_key = self._get_registry_key(session_id)
            counter_key = self._get_counter_key(session_id)
            
            registry = redis_service.get(registry_key) or {}
            counter = redis_service.get(counter_key) or 0
            
            # Get all source data
            sources = {}
            for citation_id_str, summary in registry.items():
                citation_id = int(citation_id_str)
                source_data = self.get_source_by_citation_id(session_id, citation_id)
                if source_data:
                    sources[citation_id_str] = source_data
            
            return {
                "connected": True,
                "session_id": session_id,
                "total_citations": counter,
                "registry_summary": registry,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error getting session citations: {str(e)}")
            return {"connected": False, "error": str(e)}
    
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
            pattern = f"{self.registry_prefix}{session_id}:citations:*"
            deleted_count = redis_service.delete_pattern(pattern)
            logger.info(f"Cleared {deleted_count} citation entries for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing session citations: {str(e)}")
            return False
    
    def _fallback_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback method when Redis is not available.
        
        Args:
            sources: Original sources
            
        Returns:
            Sources with simple sequential IDs
        """
        fallback_sources = []
        for i, source in enumerate(sources):
            if isinstance(source, dict):
                fallback_source = source.copy()
                fallback_source.update({
                    'citation_id': i + 1,
                    'display_id': str(i + 1),
                    'session_id': 'fallback',
                    'hash': f'fallback_{i + 1}'
                })
                fallback_sources.append(fallback_source)
        
        return fallback_sources
    
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
            all_citations = self.get_all_session_citations(session_id)
            
            if all_citations.get("connected"):
                return {
                    "connected": True,
                    "session_id": session_id,
                    "total_citations": all_citations.get("total_citations", 0),
                    "unique_sources": len(all_citations.get("sources", {})),
                    "registry_size": len(all_citations.get("registry_summary", {}))
                }
            else:
                return all_citations
                
        except Exception as e:
            logger.error(f"Error getting citation stats: {str(e)}")
            return {"connected": False, "error": str(e)}


# Create a singleton instance
session_citation_registry = SessionCitationRegistry()

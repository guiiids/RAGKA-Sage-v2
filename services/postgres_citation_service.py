"""
PostgreSQL Citation Service for RAGKA

This module provides session-scoped citation storage using PostgreSQL,
matching the pattern used by session_memory.py
"""

import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from db_manager import DatabaseManager

# Configure logging
logger = logging.getLogger(__name__)

class PostgresCitationService:
    """
    Service for managing session-scoped citations in PostgreSQL.
    
    Stores citations with sequential numbering per session, ensuring
    consistent citation IDs across all messages in a session.
    """
    
    def __init__(self):
        """Initialize the PostgreSQL citation service."""
        self._ensure_tables()
        logger.info("PostgreSQL citation service initialized")
    
    def _ensure_tables(self) -> None:
        """Create citation tables if they don't exist."""
        citation_table = """
            CREATE TABLE IF NOT EXISTS session_citations (
                session_id TEXT NOT NULL,
                citation_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                url TEXT DEFAULT '',
                source_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (session_id, citation_id)
            )
        """
        
        citation_index = """
            CREATE INDEX IF NOT EXISTS idx_session_citations_session_id 
            ON session_citations (session_id, created_at DESC)
        """
        
        citation_lookup = """
            CREATE TABLE IF NOT EXISTS session_citation_lookup (
                session_id TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                citation_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (session_id, source_hash)
            )
        """
        
        lookup_index = """
            CREATE INDEX IF NOT EXISTS idx_session_citation_lookup_session_hash
            ON session_citation_lookup (session_id, source_hash)
        """
        
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(citation_table)
                cur.execute(citation_index)
                cur.execute(citation_lookup)
                cur.execute(lookup_index)
                conn.commit()
                logger.debug("Citation tables ensured")
        except Exception as e:
            logger.error(f"Failed to create citation tables: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _generate_source_hash(self, source: Dict[str, Any]) -> str:
        """
        Generate a consistent hash for source deduplication.
        
        Args:
            source: Source dictionary containing title and content
            
        Returns:
            Hash string for the source
        """
        content = source.get('content', '')
        title = source.get('title', '')
        hash_input = f"{title}_{content[:500]}"
        
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()[:12]
    
    def _get_next_citation_id(self, session_id: str) -> int:
        """
        Get the next citation ID for the session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Next citation ID
        """
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(citation_id), 0) + 1 FROM session_citations WHERE session_id = %s",
                    (session_id,)
                )
                next_id = cur.fetchone()[0]
                return next_id
        except Exception as e:
            logger.error(f"Error getting next citation ID: {e}")
            return 1
        finally:
            conn.close()
    
    def register_sources(self, session_id: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Register sources in the session citation registry and return sources with citation IDs.
        
        Args:
            session_id: Session identifier
            sources: List of source dictionaries
            
        Returns:
            List of sources with assigned citation IDs
        """
        if not sources:
            return []
        
        try:
            registered_sources = []
            conn = DatabaseManager.get_connection()
            
            try:
                with conn.cursor() as cur:
                    for source in sources:
                        if not isinstance(source, dict):
                            continue
                        
                        # Generate hash for this source
                        source_hash = self._generate_source_hash(source)
                        
                        # Check if source already exists
                        cur.execute(
                            "SELECT citation_id FROM session_citation_lookup WHERE session_id = %s AND source_hash = %s",
                            (session_id, source_hash)
                        )
                        existing = cur.fetchone()
                        
                        if existing:
                            # Source already exists, reuse citation ID
                            citation_id = existing[0]
                            logger.debug(f"Reusing citation ID {citation_id} for existing source")
                        else:
                            logger.debug(f"Inserting new citation ID {citation_id} for source with title: {source.get('title', '')}")
                            # New source, assign new citation ID
                            citation_id = self._get_next_citation_id(session_id)
                            
                            # Store the full source data
                            cur.execute(
                                """INSERT INTO session_citations 
                                   (session_id, citation_id, title, content, url, source_hash) 
                                   VALUES (%s, %s, %s, %s, %s, %s)""",
                                (
                                    session_id, 
                                    citation_id,
                                    source.get('title', f'Source {citation_id}'),
                                    source.get('content', ''),
                                    source.get('url', ''),
                                    source_hash
                                )
                            )
                            
                            # Store the lookup mapping
                            cur.execute(
                                """INSERT INTO session_citation_lookup 
                                   (session_id, source_hash, citation_id) 
                                   VALUES (%s, %s, %s)""",
                                (session_id, source_hash, citation_id)
                            )
                            
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
                
                conn.commit()
                logger.info(f"Registered {len(registered_sources)} sources for session {session_id}")
                return registered_sources
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error registering sources: {e}")
            return self._fallback_sources(sources)
    
    def get_source_by_citation_id(self, session_id: str, citation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get source data by citation ID.
        
        Args:
            session_id: Session identifier
            citation_id: Citation ID to look up
            
        Returns:
            Source data if found, None otherwise
        """
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT citation_id, title, content, url, source_hash 
                       FROM session_citations 
                       WHERE session_id = %s AND citation_id = %s""",
                    (session_id, citation_id)
                )
                row = cur.fetchone()
                
                if row:
                    return {
                        'citation_id': row[0],
                        'title': row[1],
                        'content': row[2],
                        'url': row[3],
                        'hash': row[4],
                        'id': f'source_{row[0]}',
                        'display_id': str(row[0])
                    }
                else:
                    logger.debug(f"No source found for citation ID {citation_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving source by citation ID: {e}")
            return None
        finally:
            conn.close()
    
    def get_all_session_citations(self, session_id: str) -> Dict[str, Any]:
        """
        Get all citations for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with all session citations
        """
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT citation_id, title, content, url, source_hash, created_at
                       FROM session_citations 
                       WHERE session_id = %s 
                       ORDER BY citation_id""",
                    (session_id,)
                )
                rows = cur.fetchall()
                
                sources = {}
                for row in rows:
                    citation_id = str(row[0])
                    sources[citation_id] = {
                        'citation_id': row[0],
                        'title': row[1],
                        'content': row[2],
                        'url': row[3],
                        'hash': row[4],
                        'created_at': row[5].isoformat() if row[5] else None,
                        'id': f'source_{row[0]}',
                        'display_id': citation_id
                    }
                
                return {
                    "connected": True,
                    "session_id": session_id,
                    "total_citations": len(sources),
                    "sources": sources
                }
                
        except Exception as e:
            logger.error(f"Error getting session citations: {e}")
            return {"connected": False, "error": str(e)}
        finally:
            conn.close()
    
    def clear_session_citations(self, session_id: str) -> bool:
        """
        Clear all citations for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM session_citations WHERE session_id = %s", (session_id,))
                cur.execute("DELETE FROM session_citation_lookup WHERE session_id = %s", (session_id,))
                conn.commit()
                logger.info(f"Cleared all citations for session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing session citations: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_citation_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get citation statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with citation statistics
        """
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM session_citations WHERE session_id = %s",
                    (session_id,)
                )
                total_citations = cur.fetchone()[0]
                
                cur.execute(
                    "SELECT COUNT(DISTINCT source_hash) FROM session_citations WHERE session_id = %s",
                    (session_id,)
                )
                unique_sources = cur.fetchone()[0]
                
                return {
                    "connected": True,
                    "session_id": session_id,
                    "total_citations": total_citations,
                    "unique_sources": unique_sources
                }
                
        except Exception as e:
            logger.error(f"Error getting citation stats: {e}")
            return {"connected": False, "error": str(e)}
        finally:
            conn.close()
    
    def _fallback_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback method when database is not available.
        
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


# Create a singleton instance
postgres_citation_service = PostgresCitationService()

"""
Simple Conversation Manager for RAGKA

This replaces the over-engineered entity detection system with simple Redis-based
conversation memory that GPT-4 can understand naturally.
"""

import logging
import time
from typing import Dict, List, Optional
from services.redis_service import redis_service

logger = logging.getLogger(__name__)

class SimpleConversationManager:
    """
    Simple conversation manager that stores query/response pairs in Redis.
    
    This is the complete replacement for:
    - conversation_intelligence.py
    - intelligent_classifier.py  
    - query_mediator.py
    - All entity detection logic
    """
    
    def __init__(self):
        self.redis = redis_service
        self.conversation_ttl = 3600  # 1 hour
        self.max_history_length = 10  # Keep last 10 exchanges
        
        logger.info("Simple Conversation Manager initialized")
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        
        Returns:
            List of {"query": str, "response": str, "timestamp": float} dicts
        """
        try:
            history = self.redis.get(f"conversation:{session_id}:history") or []
            return history
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def add_exchange(
        self, 
        session_id: str, 
        query: str, 
        response: str
    ) -> None:
        """
        Add a query/response exchange to conversation history.
        
        Args:
            session_id: Session ID
            query: User's query
            response: Bot's response
        """
        try:
            history = self.get_conversation_history(session_id)
            
            # Add new exchange
            exchange = {
                "query": query,
                "response": response,
                "timestamp": time.time()
            }
            history.append(exchange)
            
            # Keep only recent exchanges
            if len(history) > self.max_history_length:
                history = history[-self.max_history_length:]
            
            # Store back to Redis
            self.redis.set(
                f"conversation:{session_id}:history", 
                history, 
                self.conversation_ttl
            )
            
            logger.info(f"Added exchange to conversation history for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error adding exchange to conversation: {e}")
    
    def build_context_prompt(self, session_id: str) -> str:
        """
        Build context prompt from conversation history.
        
        Returns:
            Context string to include in GPT-4 prompt
        """
        try:
            history = self.get_conversation_history(session_id)
            
            if not history:
                return ""
            
            # Build context from recent exchanges
            context_parts = ["Previous conversation:"]
            
            for exchange in history[-5:]:  # Last 5 exchanges
                context_parts.append(f"User: {exchange['query']}")
                # Truncate long responses for context
                response = exchange['response'][:200] + "..." if len(exchange['response']) > 200 else exchange['response']
                context_parts.append(f"Assistant: {response}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building context prompt: {e}")
            return ""
    
    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        try:
            self.redis.delete(f"conversation:{session_id}:history")
            logger.info(f"Cleared conversation history for session {session_id}")
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
    
    def get_session_summary(self, session_id: str) -> Dict:
        """
        Get a summary of the session for debugging/monitoring.
        
        Returns:
            Dict with session info
        """
        try:
            history = self.get_conversation_history(session_id)
            
            if not history:
                return {"exchanges": 0, "topics": [], "last_activity": None}
            
            return {
                "exchanges": len(history),
                "topics": self._extract_topics(history),
                "last_activity": history[-1]["timestamp"] if history else None,
                "session_duration": history[-1]["timestamp"] - history[0]["timestamp"] if len(history) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting session summary: {e}")
            return {"exchanges": 0, "topics": [], "last_activity": None}
    
    def _extract_topics(self, history: List[Dict]) -> List[str]:
        """
        Extract main topics from conversation history.
        Simple extraction based on capitalized words in queries.
        """
        topics = set()
        
        for exchange in history:
            # Look for capitalized words that might be topics
            words = exchange["query"].split()
            for word in words:
                # Simple heuristic: capitalized words > 2 chars, not at start of sentence
                clean_word = word.strip("?.,!").strip()
                if (len(clean_word) > 2 and 
                    clean_word[0].isupper() and
                    clean_word not in ['What', 'How', 'Where', 'When', 'Why', 'Which', 'Who']):
                    topics.add(clean_word)
        
        return list(topics)[:5]  # Return top 5 topics


# Create singleton instance
simple_conversation_manager = SimpleConversationManager()

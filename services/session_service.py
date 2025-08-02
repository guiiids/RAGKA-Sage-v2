"""
Session Service Module

This module handles session management for RAG assistants,
maintaining separate conversation contexts for each user session.
"""

import logging
from rag_assistant_v2 import FlaskRAGAssistant

logger = logging.getLogger(__name__)

# Dictionary to store RAG assistant instances by session ID
rag_assistants = {}


def get_rag_assistant(session_id):
    """
    Get or create a RAG assistant for the given session ID.
    
    Args:
        session_id (str): The unique session identifier
        
    Returns:
        FlaskRAGAssistant: The RAG assistant instance for this session
    """
    if session_id not in rag_assistants:
        logger.info(f"Creating new RAG assistant for session {session_id}")
        rag_assistants[session_id] = FlaskRAGAssistant()
    return rag_assistants[session_id]


def clear_session(session_id):
    """
    Clear the RAG assistant for a specific session.
    
    Args:
        session_id (str): The session ID to clear
        
    Returns:
        bool: True if session was cleared, False if session didn't exist
    """
    if session_id in rag_assistants:
        logger.info(f"Clearing RAG assistant for session {session_id}")
        del rag_assistants[session_id]
        return True
    return False


def get_active_sessions():
    """
    Get a list of all active session IDs.
    
    Returns:
        list: List of active session IDs
    """
    return list(rag_assistants.keys())


def get_session_count():
    """
    Get the number of active sessions.
    
    Returns:
        int: Number of active sessions
    """
    return len(rag_assistants)

"""
Logging configuration for the RAG improvement implementation
"""
import logging
import os
from logging.handlers import RotatingFileHandler
import sys
import json
import uuid
from datetime import datetime
import threading

def setup_improvement_logging():
    """
    Set up dedicated logging for the RAG improvement process.
    Creates a separate log file and configures formatters.
    """
    # Create logger
    logger = logging.getLogger('rag_improvement')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Clear default root handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Prevent logs from propagating to the root logger
    logger.propagate = False
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create file handler for improvement logs
    file_handler = RotatingFileHandler(
        'logs/rag_improvement_logs.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter for file handler
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    return logger

def get_phase_logger(phase_number):
    """
    Get a logger for a specific implementation phase.
    
    Args:
        phase_number: The phase number (1-6)
        
    Returns:
        A logger with phase-specific prefix
    """
    logger = logging.getLogger(f'rag_improvement.phase{phase_number}')
    
    # Create a filter to add phase prefix
    class PhaseFilter(logging.Filter):
        def filter(self, record):
            record.msg = f"[PHASE-{phase_number}] {record.msg}"
            return True
    
    # Add filter to logger
    for handler in logger.handlers:
        handler.addFilter(PhaseFilter())
    
    return logger

def get_checkpoint_logger(checkpoint_number):
    """
    Get a logger for a specific checkpoint.
    
    Args:
        checkpoint_number: The checkpoint number (1-6)
        
    Returns:
        A logger with checkpoint-specific prefix
    """
    logger = logging.getLogger(f'rag_improvement.checkpoint{checkpoint_number}')
    
    # Create a filter to add checkpoint prefix
    class CheckpointFilter(logging.Filter):
        def filter(self, record):
            record.msg = f"[CHECKPOINT-{checkpoint_number}] {record.msg}"
            return True
    
    # Add filter to logger
    for handler in logger.handlers:
        handler.addFilter(CheckpointFilter())
    
    return logger

def get_test_logger():
    """
    Get a logger for test results.
    
    Returns:
        A logger with test-specific prefix
    """
    logger = logging.getLogger('rag_improvement.test')
    
    # Create a filter to add test prefix
    class TestFilter(logging.Filter):
        def filter(self, record):
            record.msg = f"[TEST] {record.msg}"
            return True
    
    # Add filter to logger
    for handler in logger.handlers:
        handler.addFilter(TestFilter())
    
    return logger

def get_compare_logger():
    """
    Get a logger for comparison results.
    
    Returns:
        A logger with comparison-specific prefix
    """
    logger = logging.getLogger('rag_improvement.compare')
    
    # Create a filter to add comparison prefix
    class CompareFilter(logging.Filter):
        def filter(self, record):
            record.msg = f"[COMPARE] {record.msg}"
            return True
    
    # Add filter to logger
    for handler in logger.handlers:
        handler.addFilter(CompareFilter())
    
# Lock for thread-safe logging of interactions
_interaction_log_lock = threading.Lock()

def log_interaction(user_query: str, bot_response: str, tokens_inferred: int, tokens_completion: int, feedback_provided: bool, unique_id: str = None, timestamp: str = None):
    """
    Log detailed interaction information as a structured JSON entry.

    Args:
        user_query (str): The user's query.
        bot_response (str): The bot's response.
        tokens_inferred (int): Number of tokens inferred.
        tokens_completion (int): Number of tokens in completion.
        feedback_provided (bool): Whether feedback was provided.
        unique_id (str, optional): Unique ID for the interaction. Generated if not provided.
        timestamp (str, optional): ISO 8601 timestamp. Generated if not provided.
    """
    if unique_id is None:
        unique_id = str(uuid.uuid4())
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"

    log_entry = {
        "unique_id": unique_id,
        "timestamp": timestamp,
        "user_query": user_query,
        "bot_response": bot_response,
        "tokens_inferred": tokens_inferred,
        "tokens_completion": tokens_completion,
        "total_tokens": tokens_inferred + tokens_completion,
        "feedback_provided": "yes" if feedback_provided else "no"
    }

    log_json = json.dumps(log_entry)

    with _interaction_log_lock:
        main_logger.info(log_json)
    return logger

# Initialize the main logger
main_logger = setup_improvement_logging()

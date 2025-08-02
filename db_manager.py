from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import logging
import json
from datetime import datetime, timezone
from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_SSL_MODE
)
from services.utils import truncate

logger = logging.getLogger()

class DatabaseManager:
    """Handles database connections and operations for the feedback system."""
    
    @staticmethod
    def get_connection():
        """Create and return a database connection."""
        try:
            logger.debug(f"Connecting to PostgreSQL: {POSTGRES_USER}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                sslmode=POSTGRES_SSL_MODE
            )
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    @staticmethod
    def save_feedback(feedback_data):
        """Save feedback to the PostgreSQL database."""
        conn = None
        try:
            # Log the incoming feedback data for debugging
            logger.debug(f"Saving feedback data: {feedback_data}")
            
            # Extract the user query and bot response
            # The frontend might be sending different keys than what we expect
            user_query = feedback_data.get("question", "")
            bot_response = feedback_data.get("response", "")
            
            # If user_query and bot_response are empty, try to get them from other fields
            # This is a fallback mechanism to handle different data structures
            if not user_query and "user_query" in feedback_data:
                user_query = feedback_data["user_query"]
            
            if not bot_response and "bot_response" in feedback_data:
                bot_response = feedback_data["bot_response"]
            
            citations = feedback_data.get("citations", [])
            
            conn = DatabaseManager.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO votes 
                    (user_query, bot_response, evaluation_json, feedback_tags, comment, citations)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING vote_id
                    """,
                    (
                        user_query,
                        bot_response,
                        Json(feedback_data.get("evaluation_json", {})),
                        feedback_data["feedback_tags"],
                        feedback_data.get("comment", ""),
                        Json(citations)
                    )
                )
                vote_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Feedback saved successfully with ID: {vote_id}")
                return vote_id
        except Exception as e:
            logger.error(f"Error saving feedback to database: {e}")
            raise
        finally:
            if conn is not None:
                conn.close()
    
    @staticmethod
    def get_feedback_summary(start_date=None, end_date=None):
        """Get summary statistics of collected feedback, optionally filtered by date range."""
        import logging
        conn = None
        try:
            logging.info(f"get_feedback_summary called with start_date={start_date}, end_date={end_date}")
            conn = DatabaseManager.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build date filter clause and parameters
                date_filter = ""
                params = []
                if start_date and end_date:
                    date_filter = "WHERE timestamp BETWEEN %s AND %s"
                    params = [start_date, end_date]
                elif start_date:
                    date_filter = "WHERE timestamp >= %s"
                    params = [start_date]
                elif end_date:
                    date_filter = "WHERE timestamp <= %s"
                    params = [end_date]

                # Count total feedback entries with optional date filter
                cursor.execute(f"SELECT COUNT(*) as total_feedback FROM votes {date_filter}", params)
                total_feedback = cursor.fetchone()["total_feedback"]
                
                # Count positive feedback (contains "Looks Good") with optional date filter
                cursor.execute(
                    f"SELECT COUNT(*) as positive_feedback FROM votes {date_filter} AND 'Looks Good / Accurate & Clear' = ANY(feedback_tags)" if date_filter else "SELECT COUNT(*) as positive_feedback FROM votes WHERE 'Looks Good / Accurate & Clear' = ANY(feedback_tags)",
                    params
                )
                positive_feedback = cursor.fetchone()["positive_feedback"]
                
                # Get recent feedback (last 5 entries) with optional date filter
                recent_query = (
                    "SELECT vote_id, user_query, feedback_tags, comment, timestamp "
                    "FROM votes "
                    f"{date_filter} "
                    "ORDER BY timestamp DESC "
                    "LIMIT 5"
                )
                cursor.execute(recent_query, params)
                recent_feedback = cursor.fetchall()
                
                summary = {
                    'total_feedback': total_feedback,
                    'positive_feedback': positive_feedback,
                    'negative_feedback': total_feedback - positive_feedback,
                    'recent_feedback': recent_feedback
                }
                
                return summary
        except Exception as e:
            logger.error(f"Error generating feedback summary: {e}")
            return {
                'total_feedback': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'recent_feedback': []
            }
        finally:
            if conn is not None:
                conn.close()
    
    @staticmethod
    def get_query_analytics(start_date=None, end_date=None):
        """Analyze query patterns and generate statistics, optionally filtered by date range."""
        import logging
        conn = None
        try:
            logging.info(f"get_query_analytics called with start_date={start_date}, end_date={end_date}")
            conn = DatabaseManager.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                date_filter = ""
                params = []
                if start_date and end_date:
                    date_filter = "WHERE timestamp BETWEEN %s AND %s"
                    params = [start_date, end_date]
                elif start_date:
                    date_filter = "WHERE timestamp >= %s"
                    params = [start_date]
                elif end_date:
                    date_filter = "WHERE timestamp <= %s"
                    params = [end_date]

                # Count total unique queries
                cursor.execute(f"SELECT COUNT(DISTINCT user_query) as total_queries FROM votes {date_filter}", params)
                total_queries = cursor.fetchone()["total_queries"]

                # Count total feedback entries
                cursor.execute(f"SELECT COUNT(*) as queries_with_feedback FROM votes {date_filter}", params)
                queries_with_feedback = cursor.fetchone()["queries_with_feedback"]

                # Count successful queries (with "Looks Good" tag)
                if date_filter:
                    cursor.execute(
                        f"SELECT COUNT(*) as successful_queries FROM votes {date_filter} AND 'Looks Good / Accurate & Clear' = ANY(feedback_tags)",
                        params
                    )
                else:
                    cursor.execute(
                        "SELECT COUNT(*) as successful_queries FROM votes WHERE 'Looks Good / Accurate & Clear' = ANY(feedback_tags)"
                    )
                successful_queries = cursor.fetchone()["successful_queries"]

                # Get recent queries
                recent_query = (
                    "SELECT user_query, timestamp "
                    "FROM votes "
                    f"{date_filter} "
                    "ORDER BY timestamp DESC "
                    "LIMIT 5"
                )
                cursor.execute(recent_query, params)
                recent_queries = cursor.fetchall()

                analytics = {
                    'total_queries': total_queries,
                    'queries_with_feedback': queries_with_feedback,
                    'successful_queries': successful_queries,
                    'recent_queries': recent_queries
                }

                return analytics
        except Exception as e:
            logging.error(f"Error generating query analytics: {e}")
            return {
                'total_queries': 0,
                'queries_with_feedback': 0,
                'successful_queries': 0,
                'recent_queries': []
            }
        finally:
            if conn is not None:
                conn.close()
    
    @staticmethod
    def log_helpee_activity(user_query: str, response_text: str, prompt_tokens: int = None, completion_tokens: int = None, total_tokens: int = None, model: str = None):
        """
        Log LLM helpee activity into helpee_logs table, including model.
        Returns the inserted helpee_log ID.
        """
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO helpee_logs
                      (user_query, response_text, prompt_tokens, completion_tokens, total_tokens, model)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (user_query, response_text, prompt_tokens, completion_tokens, total_tokens, model)
                )
                log_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Helpee activity logged successfully with ID: {log_id}")
                return log_id
        except Exception as e:
            logger.error(f"Error logging helpee activity: {e}")
            raise
        finally:
            conn.close()
    @staticmethod
    def save_helpee_log(log_data):
        """Save helpee log entry to the PostgreSQL database."""
        conn = None
        try:
            logger.debug(f"Saving helpee log data: {log_data}")

            user_query = log_data.get("user_query", "")
            response_text = log_data.get("response_text", "")
            prompt_tokens = log_data.get("prompt_tokens")
            completion_tokens = log_data.get("completion_tokens")
            total_tokens = log_data.get("total_tokens")

            conn = DatabaseManager.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO helpee_logs 
                    (user_query, response_text, prompt_tokens, completion_tokens, total_tokens)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (user_query, response_text, prompt_tokens, completion_tokens, total_tokens)
                )
                log_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Helpee log saved successfully with ID: {log_id}")
                return log_id
        except Exception as e:
            logger.error(f"Error logging helpee activity: {e}")
            raise
        finally:
            conn.close()
    @staticmethod
    def log_helpee_cost(helpee_log_id: int, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, prompt_cost: float, completion_cost: float, total_cost: float):
        """Log cost breakdown for a helpee_logs entry into helpee_costs table."""
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO helpee_costs
                      (helpee_log_id, model, prompt_tokens, completion_tokens, total_tokens, prompt_cost, completion_cost, total_cost)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (helpee_log_id, model, prompt_tokens, completion_tokens, total_tokens, prompt_cost, completion_cost, total_cost)
                )
                conn.commit()
                logger.info(f"Helpee cost logged for log_id {helpee_log_id}: total_cost={total_cost}")
        except Exception as e:
            logger.error(f"Error logging helpee cost: {e}")
            raise
        finally:
            conn.close()

    @staticmethod
    def get_helpee_costs(start_date=None, end_date=None):
        """Retrieve helpee cost entries optionally filtered by timestamp."""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                date_filter = ""
                params = []
                if start_date and end_date:
                    date_filter = "WHERE timestamp BETWEEN %s AND %s"
                    params = [start_date, end_date]
                elif start_date:
                    date_filter = "WHERE timestamp >= %s"
                    params = [start_date]
                elif end_date:
                    date_filter = "WHERE timestamp <= %s"
                    params = [end_date]
                query = f"""
                    SELECT helpee_log_id, model, prompt_tokens, completion_tokens, total_tokens, prompt_cost, completion_cost, total_cost, timestamp
                    FROM helpee_costs
                    {date_filter}
                    ORDER BY timestamp DESC
                """
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching helpee costs: {e}")
            return []
        finally:
            if conn:
                conn.close()
        
        @staticmethod
        def get_tag_distribution(start_date=None, end_date=None):
            """Get distribution of feedback tags, optionally filtered by date range."""
            import logging
            conn = None
            try:
                logging.info(f"get_tag_distribution called with start_date={start_date}, end_date={end_date}")
                conn = DatabaseManager.get_connection()
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    date_filter = ""
                    params = []
                    if start_date and end_date:
                        date_filter = "WHERE timestamp BETWEEN %s AND %s"
                        params = [start_date, end_date]
                    elif start_date:
                        date_filter = "WHERE timestamp >= %s"
                        params = [start_date]
                    elif end_date:
                        date_filter = "WHERE timestamp <= %s"
                        params = [end_date]

                    query = (
                        "SELECT unnest(feedback_tags) as tag, COUNT(*) as count "
                        "FROM votes "
                        f"{date_filter} "
                        "GROUP BY tag "
                        "ORDER BY count DESC"
                    )
                    cursor.execute(query, params)
                    tag_distribution = cursor.fetchall()
                    return tag_distribution
            except Exception as e:
                logging.error(f"Error getting tag distribution: {e}")
                return []
            finally:
                if conn is not None:
                    conn.close()
    
    @staticmethod
    def get_time_metrics(start_date=None, end_date=None):
        """Calculate response time metrics, optionally filtered by date range."""
        import logging
        logging.warning("Response time metrics cannot be calculated: missing user query timestamps.")
        # Return placeholder metrics as 'N/A' strings to clearly indicate no data
        return {
            "avg_response_time": "N/A",
            "min_response_time": "N/A",
            "max_response_time": "N/A"
        }
    
    @staticmethod
    def log_rag_query(query, response, sources, context, sql_query=None):
        """
        Log a RAG query, response, and source metadata to the database.
        
        Args:
            query (str): The user's query
            response (str): The generated response
            sources (list): List of sources used in the response
            context (str): The context used to generate the response
            sql_query (str, optional): The SQL query used to retrieve data
            
        Returns:
            int: The ID of the logged entry
        """
        conn = None
        try:
            # Prepare the data
            timestamp = datetime.now(timezone.utc)
            query = truncate(query)
            response = truncate(response)
            context = truncate(context)
            
            # Create a structured record of the sources with all available metadata
            source_metadata = []
            for source in sources:
                if isinstance(source, dict):
                    source_metadata.append(source)
                else:
                    # If source is not a dict, create a simple dict with the source as content
                    source_metadata.append({"content": str(source)})
            
            # Connect to the database
            conn = DatabaseManager.get_connection()
            with conn.cursor() as cursor:
                # Check if rag_queries table exists, create it if not
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'rag_queries'
                    );
                """)
                table_exists = cursor.fetchone()[0]
                
                if not table_exists:
                    # Create the table if it doesn't exist
                    cursor.execute("""
                        CREATE TABLE rag_queries (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                            user_query TEXT NOT NULL,
                            response TEXT NOT NULL,
                            sources JSONB NOT NULL,
                            context TEXT NOT NULL,
                            sql_query TEXT
                        );
                    """)
                    conn.commit()
                    logger.info("Created rag_queries table")
                
                # Insert the data
                cursor.execute(
                    """
                    INSERT INTO rag_queries 
                    (timestamp, user_query, response, sources, context, sql_query)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        timestamp,
                        query,
                        response,
                        Json(source_metadata),
                        context,
                        sql_query
                    )
                )
                entry_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Logged RAG query with ID: {entry_id}")
                return entry_id
        except Exception as e:
            logger.error(f"Error logging RAG query to database: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn is not None:
                conn.close()

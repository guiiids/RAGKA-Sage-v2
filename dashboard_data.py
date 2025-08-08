"""
dashboard_data.py

Exports the feedback analytics data and metric routines from feedback_dashboard_modern.py
for use by both a CLI and a Flask route.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import json
import collections
import re
import statistics
from datetime import datetime, timedelta
from pathlib import Path

# Ensure environment variables are loaded
load_dotenv()

DB_PARAMS = {
    'host': os.getenv('POSTGRES_HOST'),
    'port': os.getenv('POSTGRES_PORT'),
    'dbname': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'sslmode': os.getenv('POSTGRES_SSL_MODE', 'require')
}
LOG_PATH = os.path.join('logs', 'openai_calls.jsonl')

def get_db_connection():
    return psycopg2.connect(**DB_PARAMS)

def get_all_feedback():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    vote_id, 
                    user_query, 
                    bot_response, 
                    feedback_tags, 
                    comment, 
                    timestamp,
                    LENGTH(user_query) as query_length
                FROM votes 
                ORDER BY timestamp DESC
            """)
            feedback_data = cursor.fetchall()
            result = []
            for row in feedback_data:
                # Ensure serializable & proper format for web
                if row.get('timestamp'):
                    row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                if row.get('feedback_tags') is None:
                    row['feedback_tags'] = []
                result.append(dict(row))
            return result
    except Exception as e:
        print(f"Error fetching feedback data: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_total_queries():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(DISTINCT user_query) FROM votes;")
            result = cursor.fetchone()
            return result[0] if result else 0
    except Exception as e:
        print(f"Error fetching total queries: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_requests_per_hour():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT to_char(date_trunc('hour', timestamp), 'YYYY-MM-DD HH24:00') AS hour, COUNT(*) AS count
                FROM votes
                WHERE timestamp >= NOW() - INTERVAL '6 hours'
                GROUP BY hour ORDER BY hour
                LIMIT 6;
            """)
            return {row[0]: row[1] for row in cursor.fetchall()}
    except Exception as e:
        print(f"Error fetching requests per hour: {e}")
        return {}
    finally:
        if conn:
            conn.close()

def get_query_complexity_metrics():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    LENGTH(user_query) as query_length,
                    feedback_tags
                FROM votes
                WHERE user_query IS NOT NULL
            """)
            rows = cursor.fetchall()
            positive_lengths = []
            negative_lengths = []
            for row in rows:
                length = row['query_length']
                tags = row['feedback_tags'] or []
                is_positive = False
                if tags:
                    positive_indicators = ['good', 'accurate', 'helpful', 'clear', 'looks good']
                    if any(indicator in tag.lower() for tag in tags for indicator in positive_indicators):
                        is_positive = True
                if is_positive:
                    positive_lengths.append(length)
                else:
                    negative_lengths.append(length)
            avg_query_length = sum(row['query_length'] for row in rows) / len(rows) if rows else 0
            avg_positive_length = sum(positive_lengths) / len(positive_lengths) if positive_lengths else 0
            avg_negative_length = sum(negative_lengths) / len(negative_lengths) if negative_lengths else 0
            median_length = statistics.median(row['query_length'] for row in rows) if rows else 0
            return {
                'avg_query_length': round(avg_query_length, 1),
                'avg_positive_length': round(avg_positive_length, 1),
                'avg_negative_length': round(avg_negative_length, 1),
                'median_length': round(median_length, 1),
                'positive_count': len(positive_lengths),
                'negative_count': len(negative_lengths)
            }
    except Exception as e:
        print(f"Error calculating query complexity metrics: {e}")
        return {
            'avg_query_length': 0,
            'avg_positive_length': 0,
            'avg_negative_length': 0,
            'median_length': 0,
            'positive_count': 0,
            'negative_count': 0
        }
    finally:
        if conn:
            conn.close()

def get_feedback_response_time():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                WITH ordered_interactions AS (
                    SELECT 
                        timestamp,
                        LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
                    FROM votes
                    WHERE timestamp IS NOT NULL
                    ORDER BY timestamp
                )
                SELECT 
                    EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) as time_diff_seconds
                FROM ordered_interactions
                WHERE prev_timestamp IS NOT NULL
                    AND timestamp - prev_timestamp < INTERVAL '1 hour'
            """)
            time_diffs = [row[0] for row in cursor.fetchall() if row[0] is not None]
            if not time_diffs:
                return {
                    'avg_response_time_seconds': 0,
                    'median_response_time_seconds': 0,
                    'min_response_time_seconds': 0,
                    'max_response_time_seconds': 0
                }
            avg_time = sum(time_diffs) / len(time_diffs)
            median_time = statistics.median(time_diffs) if time_diffs else 0
            min_time = min(time_diffs) if time_diffs else 0
            max_time = max(time_diffs) if time_diffs else 0
            return {
                'avg_response_time_seconds': round(avg_time, 1),
                'median_response_time_seconds': round(median_time, 1),
                'min_response_time_seconds': round(min_time, 1),
                'max_response_time_seconds': round(max_time, 1)
            }
    except Exception as e:
        print(f"Error calculating response time metrics: {e}")
        return {
            'avg_response_time_seconds': 0,
            'median_response_time_seconds': 0,
            'min_response_time_seconds': 0,
            'max_response_time_seconds': 0
        }
    finally:
        if conn:
            conn.close()

def get_word_frequencies():
    conn = None
    word_counts = collections.Counter()
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_query, feedback_tags FROM votes;")
            rows = cursor.fetchall()
            for user_query, feedback_tags_data in rows:
                if user_query:
                    words = re.findall(r'\b\w+\b', user_query.lower())
                    word_counts.update(words)
                if not feedback_tags_data:
                    continue
                tags_to_process = []
                if isinstance(feedback_tags_data, list):
                    tags_to_process = feedback_tags_data
                elif isinstance(feedback_tags_data, str):
                    try:
                        parsed_tags = json.loads(feedback_tags_data)
                        if isinstance(parsed_tags, list):
                            tags_to_process = parsed_tags
                        else:
                            tags_to_process.append(str(parsed_tags))
                    except json.JSONDecodeError:
                        tags_to_process.append(feedback_tags_data)
                for tag in tags_to_process:
                    if isinstance(tag, str):
                        tag_words = re.findall(r'\b\w+\b', tag.lower())
                        word_counts.update(tag_words)
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                          'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 
                          'theirs', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                          'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
                          'does', 'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'of', 
                          'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 
                          'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
        for word in stop_words:
            word_counts.pop(word, None)
        most_common = dict(word_counts.most_common(50))
        return most_common
    except Exception as e:
        print(f"Error computing word frequencies: {e}")
        return {}
    finally:
        if conn:
            conn.close()

def determine_feedback_status(tags):
    if not tags:
        return {'status': 'Negative', 'class': 'badge neg'}
    positive_indicators = ['good', 'accurate', 'helpful', 'clear', 'looks good']
    if any(indicator in tag.lower() for tag in tags for indicator in positive_indicators):
        return {'status': 'Positive', 'class': 'badge pos'}
    return {'status': 'Negative', 'class': 'badge neg'}

def create_tag_badges(tags):
    if not tags:
        return '<span class="badge tag">No tags</span>'
    badges_html = []
    for tag in tags:
        tag_lower = tag.lower()
        if any(s in tag_lower for s in ["good", "accurate", "helpful"]):
            badge_class = "badge pos"
        elif any(s in tag_lower for s in ["incorrect", "wrong"]):
            badge_class = "badge neg"
        elif any(s in tag_lower for s in ["unclear", "confusing", "incomplete"]):
            badge_class = "badge tag"
        else:
            badge_class = "badge tag"
        badges_html.append(f'<span class="{badge_class}">{tag}</span>')
    return ''.join(badges_html)

def parse_openai_calls():
    tokens = []
    if not os.path.exists(LOG_PATH):
        return tokens
    try:
        with open(LOG_PATH, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    tot = entry.get('usage', {}).get('total_tokens')
                    if isinstance(tot, (int, float)):
                        tokens.append(tot)
                except (json.JSONDecodeError, AttributeError):
                    continue
        return tokens
    except Exception as e:
        print(f"Error reading OpenAI calls log: {e}")
        return []

def get_dashboard_metrics():
    feedback_data = get_all_feedback()
    total_queries = get_total_queries()
    total_feedback = len(feedback_data)
    positive_feedback_count = sum(1 for fb in feedback_data if determine_feedback_status(fb.get('feedback_tags', [])).get('status') == 'Positive')
    positive_feedback_pct = (positive_feedback_count / total_feedback * 100) if total_feedback else 0.0
    token_list = parse_openai_calls()
    avg_tokens = (sum(token_list) / len(token_list)) if token_list else 0.0
    query_complexity = get_query_complexity_metrics()
    response_time = get_feedback_response_time()
    word_freqs = get_word_frequencies()
    requests_per_hour_db = get_requests_per_hour()

    # Compose all metrics as needed for the dashboard
    return {
        'total_queries': total_queries,
        'total_feedback': total_feedback,
        'positive_feedback_count': positive_feedback_count,
        'positive_feedback_pct': positive_feedback_pct,
        'avg_tokens': avg_tokens,
        'query_complexity': query_complexity,
        'response_time': response_time,
        'word_freqs': word_freqs,
        'feedback_data': feedback_data,
        'requests_per_hour': requests_per_hour_db,
    }

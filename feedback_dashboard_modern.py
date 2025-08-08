#!/usr/bin/env python3
"""
Feedback Dashboard - Modern Version

A single-file script that connects directly to PostgreSQL and generates a dashboard
displaying all feedback data with a modern UI using Tailwind CSS components.
Features include:
- Enhanced metrics dashboard with additional analytics
- Interactive charts and visualizations
- Modern, responsive design with improved UX
- Detailed feedback analysis

Usage:
    python feedback_dashboard_modern.py

Requirements:
    - psycopg2-binary
    - python-dotenv
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
import webbrowser
from pathlib import Path
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
import html
from datetime import datetime, timedelta
import json
import collections
import re
import statistics
from typing import Dict, List, Any, Tuple, Optional, Union

# =====================================================================
# CONFIGURATION AND SETUP
# =====================================================================

# Load environment variables
load_dotenv()

# Database connection parameters
DB_PARAMS = {
    'host': os.getenv('POSTGRES_HOST'),
    'port': os.getenv('POSTGRES_PORT'),
    'dbname': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'sslmode': os.getenv('POSTGRES_SSL_MODE', 'require')
}

# Path to OpenAI calls log
LOG_PATH = os.path.join('logs', 'openai_calls.jsonl')

# =====================================================================
# DATABASE FUNCTIONS
# =====================================================================

def get_db_connection():
    """Create and return a database connection."""
    try:
        print("Attempting to connect to PostgreSQL database...")
        conn = psycopg2.connect(**DB_PARAMS)
        print("Database connection established successfully.")
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

def get_all_feedback():
    """Fetch all feedback data from the database."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
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
            """
            print("Executing query to fetch all feedback data...")
            cursor.execute(query)
            feedback_data = cursor.fetchall()
            
            result = []
            for row in feedback_data:
                if row.get('timestamp'):
                    row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                if row.get('feedback_tags') is None:
                    row['feedback_tags'] = []
                result.append(dict(row))
            
            print(f"Retrieved {len(result)} feedback records.")
            return result
    except Exception as e:
        print(f"Error fetching feedback data: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_total_queries():
    """Fetch total number of distinct user queries from the database."""
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
    """Fetch count of requests grouped by hour from the database for the last 6 hours."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Explicitly limit to exactly 6 hours to prevent browser crashes
            cursor.execute("""
                SELECT to_char(date_trunc('hour', timestamp), 'YYYY-MM-DD HH24:00') AS hour, COUNT(*) AS count
                FROM votes
                WHERE timestamp >= NOW() - INTERVAL '6 hours'
                GROUP BY hour ORDER BY hour
                LIMIT 6;
            """)
            result = {row[0]: row[1] for row in cursor.fetchall()}
            print(f"Retrieved requests per hour data: {result}")
            return result
    except Exception as e:
        print(f"Error fetching requests per hour: {e}")
        return {}
    finally:
        if conn:
            conn.close()

def get_query_complexity_metrics():
    """
    Analyze query complexity and its correlation with feedback sentiment.
    Returns metrics about query length and its relationship to feedback.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get query lengths and associated feedback
            cursor.execute("""
                SELECT 
                    LENGTH(user_query) as query_length,
                    feedback_tags
                FROM votes
                WHERE user_query IS NOT NULL
            """)
            
            rows = cursor.fetchall()
            
            # Process the data
            positive_lengths = []
            negative_lengths = []
            
            for row in rows:
                length = row['query_length']
                tags = row['feedback_tags'] or []
                
                # Determine if feedback is positive
                is_positive = False
                if tags:
                    positive_indicators = ['good', 'accurate', 'helpful', 'clear', 'looks good']
                    if any(indicator in tag.lower() for tag in tags for indicator in positive_indicators):
                        is_positive = True
                
                if is_positive:
                    positive_lengths.append(length)
                else:
                    negative_lengths.append(length)
            
            # Calculate metrics
            avg_query_length = sum(row['query_length'] for row in rows) / len(rows) if rows else 0
            avg_positive_length = sum(positive_lengths) / len(positive_lengths) if positive_lengths else 0
            avg_negative_length = sum(negative_lengths) / len(negative_lengths) if negative_lengths else 0
            
            # Calculate median and percentiles if we have enough data
            median_length = statistics.median(row['query_length'] for row in rows) if rows else 0
            
            # Calculate correlation between length and sentiment
            correlation = {
                'avg_query_length': round(avg_query_length, 1),
                'avg_positive_length': round(avg_positive_length, 1),
                'avg_negative_length': round(avg_negative_length, 1),
                'median_length': round(median_length, 1),
                'positive_count': len(positive_lengths),
                'negative_count': len(negative_lengths)
            }
            
            print(f"Query complexity metrics calculated: {correlation}")
            return correlation
            
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
    """
    Calculate average time between consecutive interactions.
    This is an approximation of response time since we don't have exact request-response pairing.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Get timestamps ordered by user
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
                    AND timestamp - prev_timestamp < INTERVAL '1 hour'  -- Filter out likely unrelated interactions
            """)
            
            time_diffs = [row[0] for row in cursor.fetchall() if row[0] is not None]
            
            if not time_diffs:
                return {
                    'avg_response_time_seconds': 0,
                    'median_response_time_seconds': 0,
                    'min_response_time_seconds': 0,
                    'max_response_time_seconds': 0
                }
            
            # Calculate metrics
            avg_time = sum(time_diffs) / len(time_diffs)
            median_time = statistics.median(time_diffs) if time_diffs else 0
            min_time = min(time_diffs) if time_diffs else 0
            max_time = max(time_diffs) if time_diffs else 0
            
            result = {
                'avg_response_time_seconds': round(avg_time, 1),
                'median_response_time_seconds': round(median_time, 1),
                'min_response_time_seconds': round(min_time, 1),
                'max_response_time_seconds': round(max_time, 1)
            }
            
            print(f"Response time metrics calculated: {result}")
            return result
            
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
    """Compute word frequencies from user queries and feedback tags, robustly handling tag formats."""
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
                        # Safer parsing than eval()
                        parsed_tags = json.loads(feedback_tags_data)
                        if isinstance(parsed_tags, list):
                            tags_to_process = parsed_tags
                        else:
                            tags_to_process.append(str(parsed_tags))
                    except json.JSONDecodeError:
                        # Fallback for non-JSON strings (e.g., "{tag1,tag2}")
                        tags_to_process.append(feedback_tags_data)
                
                for tag in tags_to_process:
                    if isinstance(tag, str):
                        tag_words = re.findall(r'\b\w+\b', tag.lower())
                        word_counts.update(tag_words)
        
        # Remove common English stop words for a cleaner word cloud
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                          'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 
                          'theirs', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                          'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
                          'does', 'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'of', 
                          'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 
                          'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
        for word in stop_words:
            word_counts.pop(word, None)
        
        # Limit to top 50 words to prevent browser crashes
        most_common = dict(word_counts.most_common(50))
        print(f"Word cloud limited to top 50 words (from {len(word_counts)} total)")
        return most_common
    except Exception as e:
        print(f"Error computing word frequencies: {e}")
        return {}
    finally:
        if conn:
            conn.close()

# =====================================================================
# DATA PROCESSING FUNCTIONS
# =====================================================================

def determine_feedback_status(tags):
    """Determine if feedback is positive or negative based on tags."""
    if not tags:
        return {'status': 'Negative', 'class': 'badge badge-red'}
    
    positive_indicators = ['good', 'accurate', 'helpful', 'clear', 'looks good']
    if any(indicator in tag.lower() for tag in tags for indicator in positive_indicators):
        return {'status': 'Positive', 'class': 'badge badge-green'}
    
    return {'status': 'Negative', 'class': 'badge badge-red'}

def create_tag_badges(tags):
    """Create HTML for tag badges."""
    if not tags:
        return '<span class="text-gray-400">No tags</span>'
    
    badges_html = []
    for tag in tags:
        tag_lower = tag.lower()
        if any(s in tag_lower for s in ["good", "accurate", "helpful"]):
            badge_class = "badge badge-green"
        elif any(s in tag_lower for s in ["incorrect", "wrong"]):
            badge_class = "badge badge-red"
        elif any(s in tag_lower for s in ["unclear", "confusing", "incomplete"]):
            badge_class = "badge badge-yellow"
        else:
            badge_class = "badge badge-blue"
        badges_html.append(f'<span class="{badge_class}">{html.escape(tag)}</span>')
    
    return ''.join(badges_html)

def parse_openai_calls():
    """Parse the openai_calls.jsonl log file to compute total tokens per call."""
    tokens = []
    if not os.path.exists(LOG_PATH):
        print(f"Warning: OpenAI calls log file not found at {LOG_PATH}")
        return tokens
    
    try:
        with open(LOG_PATH, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    tot = entry.get('usage', {}).get('total_tokens')
                    if isinstance(tot, (int, float)):
                        tokens.append(tot)
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Warning: Error parsing log line: {e}")
                    continue
        
        print(f"Parsed {len(tokens)} token entries from OpenAI calls log")
        return tokens
    except Exception as e:
        print(f"Error reading OpenAI calls log: {e}")
        return []

# =====================================================================
# HTML GENERATION FUNCTIONS
# =====================================================================

def generate_table_rows(feedback_data):
    """Generate HTML table rows for feedback data."""
    if not feedback_data:
        return '<tr><td colspan="6" class="px-6 py-4 text-center text-gray-500">No feedback data available.</td></tr>'
    
    rows_html = []
    for feedback in feedback_data:
        user_query = html.escape(feedback.get('user_query', ''))
        bot_response = html.escape(feedback.get('bot_response', ''))
        comment = html.escape(feedback.get('comment', '') or '')
        timestamp = feedback.get('timestamp', '')
        tags = feedback.get('feedback_tags', [])
        
        status = determine_feedback_status(tags)
        status_badge = f'<span class="{status["class"]}">{status["status"]}</span>'
        tag_badges = create_tag_badges(tags)
        
        row = f"""
            <tr class="hover:bg-gray-50">
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{timestamp}</td>
                <td class="px-6 py-4 text-sm text-gray-500"><div class="truncate-text" title="{user_query}">{user_query}</div></td>
                <td class="px-6 py-4 text-sm text-gray-500"><div class="truncate-text" title="{bot_response}">{bot_response}</div></td>
                <td class="px-6 py-4 text-sm">{status_badge}</td>
                <td class="px-6 py-4 text-sm">{tag_badges}</td>
                <td class="px-6 py-4 text-sm text-gray-500"><div class="truncate-text" title="{comment}">{comment}</div></td>
            </tr>
        """
        rows_html.append(row)
    
    return ''.join(rows_html)

def generate_metrics_summary_html(metrics):
    """Generate the HTML for the metrics summary section with modern Tailwind UI components."""
    # Define SVG icons for different metrics
    icons = {
        'total': '<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" /></svg>',
        'positive': '<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" /></svg>',
        'time': '<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>',
        'complexity': '<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>',
        'tokens': '<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>'
    }
    
    # Create modern stat cards
    metrics_html = f"""
    <!-- Main metrics row -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <!-- Total Queries Card -->
        <div class="bg-white dark:bg-black text-white rounded-xl shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col h-full">
            <div class="p-5 flex-grow">
                <div class="flex items-center">
                    <div class="flex-shrink-0 bg-blue-500 rounded-md p-3 text-white">
                        {icons['total']}
                    </div>
                    <div class="ml-5">
                        <p class="text-sm font-medium text-gray-500 truncate">Total Queries</p>
                        <p class="mt-1 text-3xl font-semibold text-gray-900">{metrics['total_queries']}</p>
                    </div>
                </div>
            </div>
            <div class="bg-blue-50 px-5 py-3 mt-auto">
                <div class="text-sm text-blue-600">
                    <span>Unique user questions</span>
                </div>
            </div>
        </div>
        
        <!-- Total Feedback Card -->
        <div class="bg-white dark:bg-black text-white rounded-xl shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col h-full">
            <div class="p-5 flex-grow">
                <div class="flex items-center">
                    <div class="flex-shrink-0 bg-indigo-500 rounded-md p-3 text-white">
                        {icons['total']}
                    </div>
                    <div class="ml-5">
                        <p class="text-sm font-medium text-gray-500 truncate">Total Feedback</p>
                        <p class="mt-1 text-3xl font-semibold text-gray-900">{metrics['total_feedback']}</p>
                    </div>
                </div>
            </div>
            <div class="bg-indigo-50 px-5 py-3 mt-auto">
                <div class="text-sm text-indigo-600">
                    <span>All feedback entries</span>
                </div>
            </div>
        </div>
        
        <!-- Positive Feedback Card -->
        <div class="bg-white dark:bg-black text-white rounded-xl shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col h-full">
            <div class="p-5 flex-grow">
                <div class="flex items-center">
                    <div class="flex-shrink-0 bg-green-500 rounded-md p-3 text-white">
                        {icons['positive']}
                    </div>
                    <div class="ml-5">
                        <p class="text-sm font-medium text-gray-500 truncate">Positive Feedback</p>
                        <p class="mt-1 text-3xl font-semibold text-gray-900">{metrics['positive_feedback_pct']:.1f}%</p>
                    </div>
                </div>
            </div>
            <div class="bg-green-50 px-5 py-3 mt-auto">
                <div class="text-sm text-green-600">
                    <span>{metrics['positive_feedback_count']} of {metrics['total_feedback']} responses</span>
                </div>
            </div>
        </div>
        
        <!-- Avg Tokens Card -->
        <div class="bg-white dark:bg-black text-white rounded-xl shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col h-full">
            <div class="p-5 flex-grow">
                <div class="flex items-center">
                    <div class="flex-shrink-0 bg-purple-500 rounded-md p-3 text-white">
                        {icons['tokens']}
                    </div>
                    <div class="ml-5">
                        <p class="text-sm font-medium text-gray-500 truncate">Avg. Tokens</p>
                        <p class="mt-1 text-3xl font-semibold text-gray-900">{metrics['avg_tokens']:.1f}</p>
                    </div>
                </div>
            </div>
            <div class="bg-purple-50 px-5 py-3 mt-auto">
                <div class="text-sm text-purple-600">
                    <span>Per API call</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Additional metrics row -->
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        <!-- Response Time Card -->
        <div class="bg-white dark:bg-black text-white rounded-xl shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col h-full">
            <div class="p-5 flex-grow">
                <div class="flex items-center">
                    <div class="flex-shrink-0 bg-amber-500 rounded-md p-3 text-white">
                        {icons['time']}
                    </div>
                    <div class="ml-5">
                        <p class="text-sm font-medium text-gray-500 truncate">Avg. Response Time</p>
                        <p class="mt-1 text-3xl font-semibold text-gray-900">{metrics['response_time']['avg_response_time_seconds']:.1f}s</p>
                    </div>
                </div>
                <div class="mt-4">
                    <div class="flex justify-between text-sm">
                        <span class="text-gray-500">Min: {metrics['response_time']['min_response_time_seconds']:.1f}s</span>
                        <span class="text-gray-500">Median: {metrics['response_time']['median_response_time_seconds']:.1f}s</span>
                        <span class="text-gray-500">Max: {metrics['response_time']['max_response_time_seconds']:.1f}s</span>
                    </div>
                </div>
            </div>
            <div class="bg-amber-50 px-5 py-3 mt-auto">
                <div class="text-sm text-amber-600">
                    <span>Time between interactions</span>
                </div>
            </div>
        </div>
        
        <!-- Query Complexity Card -->
        <div class="bg-white dark:bg-black text-white rounded-xl shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col h-full">
            <div class="p-5 flex-grow">
                <div class="flex items-center">
                    <div class="flex-shrink-0 bg-teal-500 rounded-md p-3 text-white">
                        {icons['complexity']}
                    </div>
                    <div class="ml-5">
                        <p class="text-sm font-medium text-gray-500 truncate">Query Complexity</p>
                        <p class="mt-1 text-3xl font-semibold text-gray-900">{metrics['query_complexity']['avg_query_length']:.1f}</p>
                    </div>
                </div>
                <div class="mt-4 grid grid-cols-2 gap-4">
                    <div class="bg-green-50 rounded-lg p-2 text-center">
                        <p class="text-xs text-gray-500">Positive Queries</p>
                        <p class="text-sm font-medium text-green-600">{metrics['query_complexity']['avg_positive_length']:.1f} chars</p>
                    </div>
                    <div class="bg-red-50 rounded-lg p-2 text-center">
                        <p class="text-xs text-gray-500">Negative Queries</p>
                        <p class="text-sm font-medium text-red-600">{metrics['query_complexity']['avg_negative_length']:.1f} chars</p>
                    </div>
                </div>
            </div>
            <div class="bg-teal-50 px-5 py-3 mt-auto">
                <div class="text-sm text-teal-600">
                    <span>Average query length in characters</span>
                </div>
            </div>
        </div>
    </div>
    """
    
    return metrics_html

def generate_dashboard_html(feedback_data, metrics):
    """Generate complete HTML for the dashboard with modern UI."""
    table_rows = generate_table_rows(feedback_data)
    metrics_summary_html = generate_metrics_summary_html(metrics)
    
    # Ensure all of the last 6 hours are represented in the chart data
    now = datetime.now()
    requests_per_hour_db = get_requests_per_hour()
    chart_labels = [(now - timedelta(hours=i)).strftime('%Y-%m-%d %H:00') for i in range(5, -1, -1)]
    chart_counts = [requests_per_hour_db.get(label, 0) for label in chart_labels]
    requests_per_hour_labels_json = json.dumps(chart_labels)
    requests_per_hour_counts_json = json.dumps(chart_counts)
    
    # Get word frequencies for word cloud
    word_freqs = get_word_frequencies()
    word_cloud_data_json = json.dumps(list(word_freqs.items()))
    
    # Modern HTML template with enhanced UI
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Feedback Analytics Dashboard</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    fontFamily: {{
                        'sans': ['Inter', 'sans-serif'],
                    }},
                    colors: {{
                        'primary': {{
                            50: '#f0f9ff',
                            100: '#e0f2fe',
                            200: '#bae6fd',
                            300: '#7dd3fc',
                            400: '#38bdf8',
                            500: '#0ea5e9',
                            600: '#0284c7',
                            700: '#0369a1',
                            800: '#075985',
                            900: '#0c4a6e',
                        }}
                    }}
                }}
            }}
        }}
    </script>
    <style>
        .truncate-text {{
            max-width: 250px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        @media (max-width: 1024px) {{ .truncate-text {{ max-width: 200px; }} }}
        @media (max-width: 768px) {{ .truncate-text {{ max-width: 150px; }} }}
        
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
            line-height: 1;
            margin-right: 0.5rem;
            margin-bottom: 0.25rem;
            white-space: nowrap;
        }}
        
        .badge-green {{ background-color: #d1fae5; color: #065f46; }}
        .badge-red {{ background-color: #fee2e2; color: #b91c1c; }}
        .badge-yellow {{ background-color: #fef3c7; color: #92400e; }}
        .badge-blue {{ background-color: #dbeafe; color: #1e40af; }}
        .badge-gray {{ background-color: #f3f4f6; color: #4b5563; }}
        
        /* Animations */
        .fade-in {{
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Transitions */
        .transition-all {{
            transition-property: all;
            transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
            transition-duration: 300ms;
        }}
    </style>
</head>
<body class="bg-gray-50 font-sans">
    <!-- Header with gradient background -->
    <div class="bg-gradient-to-r from-blue-600 to-indigo-700 text-white">
        <div class="container mx-auto px-4 py-6">
            <div class="flex flex-col md:flex-row justify-between items-center">
                <div class="mb-4 md:mb-0">
                  <div class="flex items-center">
<img id="nav-logo" class="h-auto max-w-sm w-auto inline-block object-cover md:h-4" alt="Logo" src="https://content.tst-34.aws.agilent.com/wp-content/uploads/2025/06/5.png"">
      </div>
                    <h1 class="text-3xl font-bold">Feedback Analytics Dashboard</h1>
                    <p class="text-blue-100 mt-1">Comprehensive analysis of user feedback data</p>
                </div>
                <div class="flex items-center space-x-4">
                    <div class="bg-white/20 backdrop-blur-sm rounded-lg px-4 py-2">
                        <p class="text-sm text-white">Generated: <span class="font-medium">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container mx-auto px-4 py-8">
        <!-- Metrics Summary Section -->
        <div class="mb-8 fade-in">
            <h2 class="text-2xl font-bold text-gray-800 mb-6">Key Metrics</h2>
            {metrics_summary_html}
        </div>

        <!-- Charts Section -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white dark:bg-black text-white p-6 rounded-xl shadow-md hover:shadow-lg transition-all fade-in">
                <h2 class="text-xl font-semibold mb-4 text-gray-800">Requests Per Hour (Last 6 Hours)</h2>
                <div style="height: 300px; max-height: 300px;">
                    <canvas id="requestsPerHourChart" class="w-full h-full"></canvas>
                </div>
            </div>
            <div class="bg-white dark:bg-black text-white p-6 rounded-xl shadow-md hover:shadow-lg transition-all fade-in">
                <h2 class="text-xl font-semibold mb-4 text-gray-800">Word Cloud of Queries and Tags</h2>
                <div style="height: 300px; max-height: 300px;">
                    <canvas id="wordCloudCanvas" class="w-full h-full"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Analytics Overview Section -->
        <div class="relative isolate overflow-hidden bg-white dark:bg-black text-white rounded-xl shadow-md mb-8 fade-in">
            <!-- Decorative background gradient element -->
            <div class="absolute -top-80 left-[max(6rem,33%)] -z-10 transform-gpu blur-3xl sm:left-1/2 md:top-20 lg:ml-20 xl:top-3 xl:ml-56" aria-hidden="true">
                <div class="aspect-[801/1036] w-[50.0625rem] bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-30" style="clip-path: polygon(63.1% 29.6%, 100% 17.2%, 76.7% 3.1%, 48.4% 0.1%, 44.6% 4.8%, 54.5% 25.4%, 59.8% 49.1%, 55.3% 57.9%, 44.5% 57.3%, 27.8% 48%, 35.1% 81.6%, 0% 97.8%, 39.3% 100%, 35.3% 81.5%, 97.2% 52.8%, 63.1% 29.6%)"></div>
            </div>
  
            <!-- Centered content area -->
            <div class="relative py-12 sm:py-16 px-6 lg:px-8">
                <!-- User's feedback analytics list, styled as a glassmorphism card -->
                <div class="bg-white/80 backdrop-blur-sm shadow-xl rounded-2xl p-6 sm:p-10 max-w-6xl mx-auto space-y-8 text-gray-800 ring-1 ring-gray-900/5">
                    <h2 class="text-3xl font-bold tracking-tight border-b border-gray-200 pb-4 text-gray-900">Feedback Analytics Overview</h2>
                    <div class="grid md:grid-cols-2 gap-8">
                        <!-- Card 1 -->
                        <div class="bg-gray-50/70 rounded-xl p-6 shadow-sm ring-1 ring-gray-900/5 hover:bg-gray-50 transition-all">
                            <h3 class="text-xl font-semibold mb-2 text-gray-900">1. Feedback Volume Over Time</h3>
                            <p class="text-sm leading-relaxed text-gray-700">Show total feedback counts per day/week/month to spot usage spikes or lulls.<br>Overlay positive vs. negative trends to see sentiment shifts over time.</p>
                        </div>
                        <!-- Card 2 -->
                        <div class="bg-gray-50/70 rounded-xl p-6 shadow-sm ring-1 ring-gray-900/5 hover:bg-gray-50 transition-all">
                            <h3 class="text-xl font-semibold mb-2 text-gray-900">2. Positive vs. Negative Ratio</h3>
                            <p class="text-sm leading-relaxed text-gray-700">A simple card showing % positive feedback this period.<br>A sparkline next to it tracking change over the last N days.</p>
                        </div>
                        <!-- Card 3 -->
                        <div class="bg-gray-50/70 rounded-xl p-6 shadow-sm ring-1 ring-gray-900/5 hover:bg-gray-50 transition-all">
                            <h3 class="text-xl font-semibold mb-2 text-gray-900">3. Tag Frequency Distribution</h3>
                            <p class="text-sm leading-relaxed text-gray-700">Count of each feedback tag (e.g. "helpful", "inaccurate", "confusing").<br>Helps identify the most common user concerns or praises.</p>
                        </div>
                        <!-- Card 4 -->
                        <div class="bg-gray-50/70 rounded-xl p-6 shadow-sm ring-1 ring-gray-900/5 hover:bg-gray-50 transition-all">
                            <h3 class="text-xl font-semibold mb-2 text-gray-900">4. Top Queries by Feedback</h3>
                            <p class="text-sm leading-relaxed text-gray-700">List the top 10 user queries that received the most feedback.<br>Show counts and average sentiment per query.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
  
        <!-- Feedback Data Table Section -->
        <div class="bg-white dark:bg-black text-white shadow-md rounded-xl overflow-hidden mb-8 fade-in">
            <div class="p-6 border-b border-gray-200">
                <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div>
                        <h2 class="text-2xl font-bold text-gray-800">All Feedback</h2>
                        <p class="text-sm text-gray-500">Total: <span id="total-count">{len(feedback_data)}</span> entries</p>
                    </div>
                    <div class="w-full md:w-auto">
                        <div class="relative">
                            <div class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                                <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                                </svg>
                            </div>
                            <input type="text" id="search-input" placeholder="Search feedback..." class="w-full md:w-64 pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent" />
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User Query</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Response</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Tags</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Comments</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white dark:bg-black text-white dark:bg-black divide-y divide-gray-200" id="feedback-table-body">
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
        
        <footer class="mt-8 text-center text-sm text-gray-500 pb-8">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Connected directly to PostgreSQL database: {DB_PARAMS.get('host', 'N/A')}</p>
        </footer>
    </div>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/wordcloud@1.2.2/src/wordcloud2.js"></script>
    <script>
        // Search functionality
        document.addEventListener('DOMContentLoaded', function() {{
            const searchInput = document.getElementById('search-input');
            const tableRows = document.querySelectorAll('#feedback-table-body tr');
            
            searchInput.addEventListener('keyup', function() {{
                const searchTerm = this.value.toLowerCase();
                
                tableRows.forEach(row => {{
                    const text = row.textContent.toLowerCase();
                    if(text.includes(searchTerm)) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }});
                
                // Update count of visible rows
                const visibleRows = document.querySelectorAll('#feedback-table-body tr[style=""]').length;
                document.getElementById('total-count').textContent = visibleRows;
            }});
        }});
      
        // Requests per hour chart
        const ctx = document.getElementById('requestsPerHourChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {requests_per_hour_labels_json},
                datasets: [{{
                    label: 'Requests per Hour',
                    data: {requests_per_hour_counts_json},
                    backgroundColor: 'rgba(59, 130, 246, 0.7)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{ 
                    y: {{ 
                        beginAtZero: true, 
                        ticks: {{ stepSize: 1 }},
                        grid: {{
                            color: 'rgba(0, 0, 0, 0.05)'
                        }}
                    }},
                    x: {{
                        grid: {{
                            display: false
                        }}
                    }}
                }},
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top',
                        labels: {{
                            font: {{
                                family: 'Inter'
                            }}
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleFont: {{
                            family: 'Inter'
                        }},
                        bodyFont: {{
                            family: 'Inter'
                        }}
                    }}
                }}
            }}
        }});

        // Word cloud with fallback to simple bar chart
        const wordCloudData = {word_cloud_data_json};
        if (wordCloudData.length > 0) {{
            // Check if WordCloud is defined
            if (typeof WordCloud !== 'undefined') {{
                try {{
                    WordCloud(document.getElementById('wordCloudCanvas'), {{
                        list: wordCloudData,
                        gridSize: 8,
                        weightFactor: 3,
                        fontFamily: 'Inter, sans-serif',
                        color: function(word, weight) {{
                            return weight > 10 ? '#3b82f6' : 
                                   weight > 5 ? '#6366f1' : 
                                   weight > 3 ? '#8b5cf6' : '#a855f7';
                        }},
                        rotateRatio: 0.5,
                        backgroundColor: '#fff',
                        shape: 'circle',
                        drawOutOfBound: false,
                        shrinkToFit: true,
                        clearCanvas: true
                    }});
                }} catch (e) {{
                    console.error("Error generating word cloud:", e);
                    createFallbackWordChart();
                }}
            }} else {{
                console.warn("WordCloud library not available, using fallback visualization");
                createFallbackWordChart();
            }}
        }}
        
        // Fallback function to create a simple bar chart of top words
        function createFallbackWordChart() {{
            const ctx = document.getElementById('wordCloudCanvas').getContext('2d');
            
            // Sort and take top 10 words for the bar chart
            const sortedData = wordCloudData.sort((a, b) => b[1] - a[1]).slice(0, 10);
            const labels = sortedData.map(item => item[0]);
            const values = sortedData.map(item => item[1]);
            
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'Top Words',
                        data: values,
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    indexAxis: 'y',  // Horizontal bar chart
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Top 10 Words (Fallback View)',
                            font: {{ size: 14 }}
                        }},
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        x: {{ 
                            beginAtZero: true,
                            grid: {{ display: false }}
                        }},
                        y: {{
                            grid: {{ display: false }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>"""
    
    return html_content

# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    """Main function to generate and display the dashboard."""
    try:
        print("\n" + "="*80)
        print("FEEDBACK DASHBOARD GENERATOR - MODERN VERSION")
        print("="*80)
        
        print("\nPhase 1: Connecting to database and retrieving data...")
        feedback_data = get_all_feedback()
        print(f"Retrieved {len(feedback_data)} feedback records.")
        
        print("\nPhase 2: Calculating metrics...")
        # Basic metrics
        total_queries = get_total_queries()
        total_feedback = len(feedback_data)
        positive_feedback_count = sum(1 for fb in feedback_data if determine_feedback_status(fb.get('feedback_tags', [])).get('status') == 'Positive')
        positive_feedback_pct = (positive_feedback_count / total_feedback * 100) if total_feedback else 0.0
        
        # Token usage metrics
        token_list = parse_openai_calls()
        avg_tokens = (sum(token_list) / len(token_list)) if token_list else 0.0
        
        # New metrics
        query_complexity = get_query_complexity_metrics()
        response_time = get_feedback_response_time()
        
        # Combine all metrics
        metrics = {
            'total_queries': total_queries,
            'total_feedback': total_feedback,
            'positive_feedback_count': positive_feedback_count,
            'positive_feedback_pct': positive_feedback_pct,
            'avg_tokens': avg_tokens,
            'query_complexity': query_complexity,
            'response_time': response_time
        }
        
        print("\nPhase 3: Generating dashboard HTML...")
        html_content = generate_dashboard_html(feedback_data, metrics)
        
        output_path = Path('feedback_dashboard_modern.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nDashboard generated at: {output_path.absolute()}")
        
        # Open in browser
        webbrowser.open(output_path.absolute().as_uri())
        print("Dashboard opened in your default web browser.")
        print("\nDashboard generation complete!")
        
    except Exception as e:
        print(f"\nAn error occurred while generating the dashboard: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n" + "="*80)
        print("FEEDBACK DASHBOARD GENERATOR - MODERN VERSION")
        print("="*80)
        
        print("\nPhase 1: Connecting to database and retrieving data...")
        feedback_data = get_all_feedback()
        print(f"Retrieved {len(feedback_data)} feedback records.")
        
        print("\nPhase 2: Calculating metrics...")
        # Basic metrics
        total_queries = get_total_queries()
        total_feedback = len(feedback_data)
        positive_feedback_count = sum(1 for fb in feedback_data if determine_feedback_status(fb.get('feedback_tags', [])).get('status') == 'Positive')
        positive_feedback_pct = (positive_feedback_count / total_feedback * 100) if total_feedback else 0.0
        
        # Token usage metrics
        token_list = parse_openai_calls()
        avg_tokens = (sum(token_list) / len(token_list)) if token_list else 0.0
        
        # New metrics
        query_complexity = get_query_complexity_metrics()
        response_time = get_feedback_response_time()
        
        # Combine all metrics
        metrics = {
            'total_queries': total_queries,
            'total_feedback': total_feedback,
            'positive_feedback_count': positive_feedback_count,
            'positive_feedback_pct': positive_feedback_pct,
            'avg_tokens': avg_tokens,
            'query_complexity': query_complexity,
            'response_time': response_time
        }
        
        print("\nPhase 3: Generating dashboard HTML...")
        html_content = generate_dashboard_html(feedback_data, metrics)
        
        output_path = Path('feedback_dashboard_modern.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nDashboard generated at: {output_path.absolute()}")
        
        # Open in browser
        webbrowser.open(output_path.absolute().as_uri())
        print("Dashboard opened in your default web browser.")
        print("\nDashboard generation complete!")
        
    except Exception as e:
        print(f"\nAn error occurred while generating the dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
    
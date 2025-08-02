"""
Originally this was main_v2.py, but it has been renamed to main.py
And it is an alternate version of the original main.py that uses the improved RAG implementation, and memory.
"""
print("From RAGKA-v1-codex, running:", __file__)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import requests
requests.packages.urllib3.disable_warnings()
requests.sessions.Session.verify = False

import traceback
from flask import Flask, request, jsonify, render_template, Response, send_from_directory, session
import json
import logging
import sys
import os
import time
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify, render_template_string, Response, send_from_directory, session

load_dotenv()


def get_sas_token():
    return os.getenv("SAS_TOKEN")

# Import the enhanced simple Redis-only RAG implementation
from rag_assistant import EnhancedSimpleRedisRAGAssistant
from db_manager import DatabaseManager
from openai import AzureOpenAI
from config import get_cost_rates
from openai_service import OpenAIService
from rag_improvement_logging import setup_improvement_logging
from services.postgres_citation_service import postgres_citation_service
from services.session_memory import PostgresSessionMemory
from services.utils import truncate

# Set up dedicated logging for the improved implementation
logger = setup_improvement_logging()

# Add a formatter that gives both UTC ISO and local log timestamp per log record
import datetime

class DualTimestampFormatter(logging.Formatter):
    def __init__(self, fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", **kwargs):
        super().__init__(fmt=fmt, datefmt=datefmt, **kwargs)

    def format(self, record):
        # Generate ISO8601 UTC with microseconds, always ending with 'Z'
        utcnow = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        iso_utc = utcnow.isoformat(timespec="microseconds").replace("+00:00", "Z")
        # Format local timestamp (asctime), level, message
        asctime = self.formatTime(record, self.datefmt)
        # Include log level
        levelname = record.levelname
        msg = record.getMessage()
        # Example format:
        # 2025-08-01T09:10:28.3831213Z 2025-08-01 09:10:28,382 - INFO - Message...
        return f"{iso_utc} {asctime} - {levelname} - {msg}"

# Add file handler with absolute path for main application logs
logs_dir = os.path.dirname(os.path.abspath('logs/main_alternate.log'))
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

file_handler = logging.FileHandler('logs/main_alternate.log')
file_handler.setLevel(logging.DEBUG)
dual_formatter = DualTimestampFormatter(datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(dual_formatter)
logger.addHandler(file_handler)

# Stream logs to stdout for visibility
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(dual_formatter)
logger.addHandler(stream_handler)

logger.info("Alternate Flask RAG application starting up with improved procedural content handling")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key-for-sessions")


# Dictionary to store RAG assistant instances by session ID
rag_assistants = {}

# Function to get or create the SIMPLE REDIS RAG assistant for a session
def get_rag_assistant(session_id):
    """Get or create a SimpleRedisRAGAssistant for the given session ID (no intelligent routing, just basic memory + search)."""
    if session_id not in rag_assistants:
        logger.info(f"Creating new SimpleRedisRAGAssistant for session {session_id}")
        rag_assistants[session_id] = EnhancedSimpleRedisRAGAssistant(
            session_id=session_id,
            memory=PostgresSessionMemory()
        )
        # Log PostgreSQL connection status
        logger.info(f"Using PostgreSQL for session {session_id}")
    return rag_assistants[session_id]
# LLM helpee helpers
PROMPT_ENHANCER_SYSTEM_MESSAGE = QUERY_ENHANCER_SYSTEM_PROMPT = """
You enhance raw end‑user questions before they go to a Retrieval‑Augmented Generation
search over an enterprise tech‑support knowledge base.

Rewrite the user's input into one concise, information‑dense query that maximises recall
while preserving intent.

Guidelines
• Keep all meaningful keywords; expand abbreviations (e.g. "OLS" → "OpenLab Software"),
  spell out error codes, add product codenames, versions, OS names, and known synonyms.
• Remove greetings, filler, personal data, profanity, or mention of the assistant.
• Infer implicit context (platform, language, API, UI area) when strongly suggested and
  state it explicitly.
• Never ask follow‑up questions. Even if the prompt is vague, make a best‑effort guess
  using typical support context.

Output format
Return exactly one line of plain text—no markdown, no extra keys:
"<your reformulated query>"

Examples
###
User: Why won't ilab let me log in?
→ iLab Operations Software login failure Azure AD SSO authentication error troubleshooting
###
User: Printer firmware bug?
→ printer firmware bug troubleshooting latest firmware update failure printhead model unspecified
###
"""

PROMPT_ENHANCER_SYSTEM_MESSAGE_2XL = """
IDENTITY and PURPOSE

You are an expert Prompt Engineer. Your task is to rewrite a short user query into a detailed, structured prompt that will guide another AI to generate a comprehensive, high-quality answer.

CORE TRANSFORMATION PRINCIPLES

1.  **Assign a Persona:** Start by assigning a relevant expert persona to the AI (e.g., "You are an expert in...").
2.  **State the Goal:** Clearly define the primary task, often as a request for a step-by-step guide or detailed explanation.
3.  **Deconstruct the Task:** Break the user's request into a numbered list of specific instructions for the AI. This should guide the structure of the final answer.
4.  **Enrich with Context:** Anticipate the user's needs by including relevant keywords, potential sub-topics, examples, or common issues that the user didn't explicitly mention.
5.  **Define the Format:** Specify the desired output format, such as Markdown, bullet points, or a professional tone, to ensure clarity and readability.

**Example of a successful transformation:**
- **Initial Query:** `troubleshooting Agilent gc`
- **Resulting Enhanced Prompt:** A detailed, multi-step markdown prompt that begins "You are an expert in troubleshooting Agilent Gas Chromatography (GC) systems..."

STEPS

1.  Carefully analyze the user's query provided in the INPUT section.
2.  Apply the CORE TRANSFORMATION PRINCIPLES to reformulate it into a comprehensive new prompt.
3.  Generate the enhanced prompt as the final output.

OUTPUT INSTRUCTIONS

- Output only the new, enhanced prompt.
- Do not include any other commentary, headers, or explanations.
- The output must be in clean, human-readable Markdown format.

INPUT

The following is the prompt you will improve: user-query
"""

def llm_helpee(input_text: str) -> str:
    """
    Sends PROMPT_ENHANCER_SYSTEM_MESSAGE to the Azure OpenAI model, logs usage into helpee_logs, and returns the AI output.
    """
    # Prepare Azure OpenAI client
    logger.debug(f"AzureOpenAI config: endpoint={os.getenv('AZURE_OPENAI_ENDPOINT')}, api_key=***masked***, api_version={os.getenv('AZURE_OPENAI_API_VERSION')}, model={os.getenv('AZURE_OPENAI_MODEL')}")
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    )
    # Debug: log full helpee payload before sending to Azure OpenAI
    logger.debug("Helpee payload: %s", {
        "model": os.getenv("AZURE_OPENAI_MODEL"),
        "messages": [
            { "role": "system", "content": PROMPT_ENHANCER_SYSTEM_MESSAGE },
            { "role": "user",   "content": input_text }
        ]
    })
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL"),
        messages=[
            { "role": "system", "content": PROMPT_ENHANCER_SYSTEM_MESSAGE },
            { "role": "user",   "content": input_text }
        ]
    )
    answer = response.choices[0].message.content
    usage = getattr(response, "usage", {})
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    logger.debug(f"User query: {input_text}")
    logger.debug(f"Enhanced query: {answer}")
    # Log to database
    log_id = DatabaseManager.log_helpee_activity(
        user_query=input_text,  # Store the original user query
        response_text=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=os.getenv("AZURE_OPENAI_MODEL")
    )
    model = os.getenv("AZURE_OPENAI_MODEL")
    rates = get_cost_rates(model)
    # The rates from get_cost_rates are already per 1M tokens (after being multiplied by 1000)
    # So we need to divide tokens by 1M to get the correct cost
    prompt_cost = prompt_tokens * rates["prompt"] / 1000000
    completion_cost = completion_tokens * rates["completion"] / 1000000
    total_cost = prompt_cost + completion_cost
    logger.debug(
        f"Cost calculation details: model={model}, "
        f"prompt_tokens={prompt_tokens}, prompt_rate={rates['prompt']}, prompt_cost={prompt_cost}, "
        f"completion_tokens={completion_tokens}, completion_rate={rates['completion']}, completion_cost={completion_cost}, "
        f"total_cost={total_cost}"
    )
    DatabaseManager.log_helpee_cost(
        helpee_log_id=log_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        total_cost=total_cost
    )
    # Instead of using a variable like grabbedInputTxt (which is undefined in this scope),
    # you should pass the input text as a function argument to llm_helpee.
    # The value from dev_eval_chat.js should be sent to the backend via an API call.

    # For now, just return the answer as before.
    return answer

def llm_helpee_2xl(input_text: str) -> str:
    """
    Sends PROMPT_ENHANCER_SYSTEM_MESSAGE_2XL to the Azure OpenAI model, logs usage into helpee_logs, and returns the AI output.
    """
    # Prepare Azure OpenAI client
    logger.debug(f"AzureOpenAI config: endpoint={os.getenv('AZURE_OPENAI_ENDPOINT')}, api_key=***masked***, api_version={os.getenv('AZURE_OPENAI_API_VERSION')}, model={os.getenv('AZURE_OPENAI_MODEL')}")
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    )
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL"),
        messages= [
            { "role": "system", "content": PROMPT_ENHANCER_SYSTEM_MESSAGE_2XL },
            { "role": "user",   "content": input_text }
        ]
    )
    answer = response.choices[0].message.content
    usage = getattr(response, "usage", {})
    latency = getattr(response, "latency", {})
    logger.debug(f"Latency: {latency}")

    usage = getattr(response, "usage", {})
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    
    # Log to database
    log_id = DatabaseManager.log_helpee_activity(
        user_query=input_text,  # Store the original user query
        response_text=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=os.getenv("AZURE_OPENAI_MODEL")
    )
    
    model = os.getenv("AZURE_OPENAI_MODEL")
    rates = get_cost_rates(model)
    # The rates from get_cost_rates are already per 1M tokens (after being multiplied by 1000)
    # So we need to divide tokens by 1M to get the correct cost
    prompt_cost = prompt_tokens * rates["prompt"] / 1000000
    completion_cost = completion_tokens * rates["completion"] / 1000000
    total_cost = prompt_cost + completion_cost
    
    DatabaseManager.log_helpee_cost(
        helpee_log_id=log_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        total_cost=total_cost
    )
    
    return answer


@app.route("/", methods=["GET"])
def index():
    logger.info("Index page accessed")
    # Generate a session ID if one doesn't exist
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
        logger.info(f"New session created: {session['session_id']}")
    
    token = get_sas_token()
    return render_template("index.html", sas_token=token)
# API endpoint for magic button query enhancement
@app.route('/api/magic_query', methods=['POST'])
def api_magic_query():
    """Accepts raw user input, sends it to llm_helpee, and returns the enhanced output."""
    data = request.get_json() or {}
    input_text = data.get('input_text', '')
    try:
        output = llm_helpee(input_text)
        # Add a flag to indicate this is an enhanced query
        return jsonify({
            'output': output,
            'is_enhanced': True
        })
    except Exception as e:
        logger.error(f"Error in api_magic_query: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/magic_query_2xl', methods=['POST'])
def api_magic_query_2xl():
    """Accepts raw user input, sends it to llm_helpee_2xl, and returns the enhanced output."""
    data = request.get_json() or {}
    input_text = data.get('input_text', '')
    try:
        output = llm_helpee_2xl(input_text)
        # Add a flag to indicate this is an enhanced query
        return jsonify({
            'output': output,
            'is_enhanced': True
        })
    except Exception as e:
        logger.error(f"Error in api_magic_query_2xl: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
        
@app.route("/api/query", methods=["POST"])
def api_query():
    start_time = time.perf_counter()
    data = request.get_json()
    logger.info("DEBUG - Incoming /api/query payload: %s", json.dumps(data))
    user_query = data.get("query", "")
    is_enhanced = data.get("is_enhanced", False)
    logger.info(f"API query received: {truncate(user_query)}")
    
    # Get the session ID
    session_id = session.get('session_id')
    if not session_id:
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
        logger.info(f"Created new session ID: {session_id}")
    
    # Extract any settings from the request
    settings = data.get("settings", {})
    logger.info(f"DEBUG - Request settings: {json.dumps(settings)}")
    
    try:
        # Get or create the RAG assistant for this session
        rag_assistant = get_rag_assistant(session_id)
        
        # Update settings if provided
        if settings:
            for key, value in settings.items():
                if hasattr(rag_assistant, key):
                    setattr(rag_assistant, key, value)
            
            # If model is updated, update the deployment name
            if "model" in settings:
                rag_assistant.deployment_name = settings["model"]
        
        logger.info(f"DEBUG - Using model: {rag_assistant.deployment_name}")
        logger.info(f"DEBUG - Temperature: {rag_assistant.temperature}")
        logger.info(f"DEBUG - Max tokens: {rag_assistant.max_completion_tokens}")
        logger.info(f"DEBUG - Top P: {rag_assistant.top_p}")
        
        html_answer, citations = rag_assistant.generate_response(user_query)
        logger.info(f"API query response generated for: {truncate(user_query)}")
        logger.info(f"DEBUG - Response length: {len(html_answer)}")

        latency = time.perf_counter() - start_time
        logger.info(
            "Query: %s | Response: %s | Citations: %d | Latency: %.2fs",
            truncate(user_query),
            truncate(html_answer),
            len(citations),
            latency,
        )

        # Log with sources for traceability
        try:
            vote_id = DatabaseManager.log_rag_query(
                query=user_query,
                response=html_answer,
                sources=citations,
                context="",
                sql_query=None
            )
            logger.info(f"RAG query logged with ID: {vote_id}")
        except Exception as log_exc:
            logger.error(f"Failed to log RAG query: {log_exc}", exc_info=True)

        return jsonify({
            "answer": html_answer,
            "sources": citations,
            "evaluation": {}
        })
    except requests.exceptions.RequestException as e:
        logger.error("Connection error in api_query", exc_info=True)
        return jsonify({"error": "Could not connect to server. Please try again later."}), 503
    except Exception as e:
        logger.error(f"Error in api_query: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/api/query/stream", methods=["POST"])
def api_query_stream():
    """Stream RAG responses for better perceived latency"""
    start_time = time.perf_counter()
    data = request.get_json()
    logger.debug(f"api_query_stream called with payload: {data}")
    user_query = data.get("query", "")
    is_enhanced = data.get("is_enhanced", False)
    
    # Get session and RAG assistant
    session_id = session.get('session_id')
    if not session_id:
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
    
    rag_assistant = get_rag_assistant(session_id)
    
    # Apply settings if provided
    settings = data.get("settings", {})
    if settings:
        for key, value in settings.items():
            if hasattr(rag_assistant, key):
                setattr(rag_assistant, key, value)
    
    def generate():
        response_text = ""
        citations = []
        try:
            for chunk in rag_assistant.stream_rag_response(user_query):
                if isinstance(chunk, str):
                    response_text += chunk
                    # Text chunk
                    yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
                elif isinstance(chunk, dict):
                    citations = chunk.get("sources", [])
                    # Metadata (sources, evaluation, etc.)
                    yield f"data: {json.dumps({'type': 'metadata', 'data': chunk})}\n\n"

            latency = time.perf_counter() - start_time
            logger.info(
                "Query: %s | Response: %s | Citations: %d | Latency: %.2fs",
                truncate(user_query),
                truncate(response_text),
                len(citations),
                latency,
            )
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    return Response(
        generate(),
        mimetype='text/plain',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable nginx buffering
        }
    )


@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    """Clear the conversation history for the current session"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in rag_assistants:
            logger.info(f"Clearing conversation history for session {session_id}")
            rag_assistants[session_id].clear_conversation_history()
            # Also clear citations from PostgreSQL for this session
            postgres_citation_service.clear_session_citations(session_id)
            return jsonify({"success": True})
        else:
            logger.warning(f"No active session found to clear history")
            return jsonify({"success": True, "message": "No active session found"})
    except Exception as e:
        logger.error(f"Error clearing conversation history: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/cache/stats", methods=["GET"])
def api_cache_stats():
    """Get cache statistics"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in rag_assistants:
            logger.info(f"Getting cache stats for session {session_id}")
            stats = rag_assistants[session_id].get_cache_stats()
            return jsonify({"success": True, "stats": stats})
        else:
            # Return Redis connection status if no active session
            connected = redis_service.is_connected()
            health = redis_service.health_check() if connected else {}
            return jsonify({
                "success": True, 
                "stats": {
                    "connected": connected,
                    "health": health,
                    "message": "No active session found"
                }
            })
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/cache/clear", methods=["POST"])
def api_clear_cache():
    """Clear the cache for the current session"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in rag_assistants:
            logger.info(f"Clearing cache for session {session_id}")
            # Get cache type from request if provided
            data = request.get_json() or {}
            cache_type = data.get("type")
            
            # Clear cache with optional type
            success = rag_assistants[session_id].clear_cache(cache_type)
            return jsonify({"success": success})
        else:
            logger.warning(f"No active session found to clear cache")
            return jsonify({"success": True, "message": "No active session found"})
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/citations/store", methods=["POST"])
def api_store_citations():
    """Store citations for a message in Redis"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', session.get('session_id', 'default'))
        message_id = data.get('message_id', '')
        sources = data.get('sources', [])
        
        if not message_id:
            return jsonify({"success": False, "error": "message_id is required"}), 400
        
        logger.debug(f"Storing {len(sources)} citations for message {message_id} in session {session_id}")
        success = redis_citation_service.store_message_citations(session_id, message_id, sources)
        
        return jsonify({"success": success})
        
    except Exception as e:
        logger.error(f"Error storing citations: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/citations/get", methods=["GET"])
def api_get_citations():
    """Get citations for a message from Redis"""
    try:
        session_id = request.args.get('session_id', session.get('session_id', 'default'))
        message_id = request.args.get('message_id', '')
        
        if not message_id:
            return jsonify({"success": False, "error": "message_id is required"}), 400
        
        logger.debug(f"Retrieving citations for message {message_id} in session {session_id}")
        citations = redis_citation_service.get_message_citations(session_id, message_id)
        
        return jsonify({"success": True, "citations": citations})
        
    except Exception as e:
        logger.error(f"Error retrieving citations: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e), "citations": []}), 500

@app.route("/api/citations/clear", methods=["POST"])
def api_clear_citations():
    """Clear citations for a session or specific message"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', session.get('session_id', 'default'))
        message_id = data.get('message_id')  # Optional - if provided, clear only this message
        
        if message_id:
            # Clear specific message citations
            logger.info(f"Clearing citations for message {message_id} in session {session_id}")
            success = redis_citation_service.clear_message_citations(session_id, message_id)
        else:
            # Clear all session citations
            logger.info(f"Clearing all citations for session {session_id}")
            success = redis_citation_service.clear_session_citations(session_id)
        
        return jsonify({"success": success})
        
    except Exception as e:
        logger.error(f"Error clearing citations: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/citations/stats", methods=["GET"])
def api_citation_stats():
    """Get citation statistics for a session"""
    try:
        session_id = request.args.get('session_id', session.get('session_id', 'default'))
        
        logger.debug(f"Getting citation stats for session {session_id}")
        stats = redis_citation_service.get_citation_stats(session_id)
        
        return jsonify({"success": True, "stats": stats})
        
    except Exception as e:
        logger.error(f"Error getting citation stats: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

# Session-wide citation registry endpoints
@app.route("/api/session-citations/register", methods=["POST"])
def api_register_session_citations():
    """Register sources in the session-wide citation registry"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', session.get('session_id', 'default'))
        sources = data.get('sources', [])
        
        if not sources:
            return jsonify({"success": False, "error": "sources are required"}), 400
        
        logger.debug(f"Registering {len(sources)} sources for session {session_id}")
        registered_sources = postgres_citation_service.register_sources(session_id, sources)
        
        return jsonify({
            "success": True, 
            "sources": registered_sources,
            "total_registered": len(registered_sources)
        })
        
    except Exception as e:
        logger.error(f"Error registering session citations: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/session-citations/get", methods=["GET"])
def api_get_session_citation():
    """Get a specific citation by ID from the session registry"""
    try:
        session_id = request.args.get('session_id', session.get('session_id', 'default'))
        citation_id = request.args.get('citation_id', '')
        
        if not citation_id:
            return jsonify({"success": False, "error": "citation_id is required"}), 400
        
        try:
            citation_id_int = int(citation_id)
        except ValueError:
            return jsonify({"success": False, "error": "citation_id must be a number"}), 400
        
        logger.debug(f"Getting citation {citation_id} for session {session_id}")
        source_data = postgres_citation_service.get_source_by_citation_id(session_id, citation_id_int)
        
        if source_data:
            return jsonify({"success": True, "source": source_data})
        else:
            return jsonify({"success": False, "error": "Citation not found"}), 404
        
    except Exception as e:
        logger.error(f"Error getting session citation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/session-citations/all", methods=["GET"])
def api_get_all_session_citations():
    """Get all citations for a session"""
    try:
        session_id = request.args.get('session_id', session.get('session_id', 'default'))
        
        logger.debug(f"Getting all citations for session {session_id}")
        all_citations = postgres_citation_service.get_all_session_citations(session_id)
        
        return jsonify({"success": True, "data": all_citations})
        
    except Exception as e:
        logger.error(f"Error getting all session citations: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/session-citations/clear", methods=["POST"])
def api_clear_session_citations():
    """Clear all citations for a session from the registry"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', session.get('session_id', 'default'))
        
        logger.info(f"Clearing session citation registry for session {session_id}")
        success = postgres_citation_service.clear_session_citations(session_id)
        
        return jsonify({"success": success})
        
    except Exception as e:
        logger.error(f"Error clearing session citations: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/session-citations/stats", methods=["GET"])
def api_session_citation_stats():
    """Get session citation registry statistics"""
    try:
        session_id = request.args.get('session_id', session.get('session_id', 'default'))
        
        logger.debug(f"Getting session citation stats for session {session_id}")
        stats = postgres_citation_service.get_citation_stats(session_id)
        
        return jsonify({"success": True, "stats": stats})
        
    except Exception as e:
        logger.error(f"Error getting session citation stats: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

# Serve static files from the 'static' folder
@app.route("/static/<path:filename>")
def serve_static(filename):
    logger.debug(f"serve_static called for static file: {filename}")
    return send_from_directory("static", filename)

# Serve static files from the 'assets' folder
@app.route("/assets/<path:filename>")
def serve_assets(filename):
    logger.debug(f"serve_assets called for asset file: {filename}")
    return send_from_directory("assets", filename)

# Feedback submission endpoint
@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json()
    logger.debug("Incoming feedback payload: %s", json.dumps(data))
    try:
        vote_id = DatabaseManager.save_feedback(data)
        logger.info(f"Feedback saved with ID: {vote_id}")
        # Fallback: write feedback to local JSON file
        with open("logs/feedback_fallback.jsonl", "a") as f:
            f.write(json.dumps({"vote_id": vote_id, **data}) + "\n")
        return jsonify({"success": True, "vote_id": vote_id})
    except Exception as e:
        logger.error("Error saving feedback: %s", str(e), exc_info=True)
        # Fallback: write raw feedback to local JSON file
        with open("logs/feedback_fallback.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Improved Flask RAG application')
    parser.add_argument('--port', type=int, default=int(os.environ.get("PORT", 5004)),
                        help='Port to run the server on (default: 5004)')
    args = parser.parse_args()
    
    port = args.port
    logger.info(f"Starting Improved Flask app on port {port}")

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

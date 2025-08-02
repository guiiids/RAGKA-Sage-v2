"""
LLM Service Module

This module handles all LLM-related operations including query enhancement
and prompt processing for the RAG application.
"""

import os
import logging
from openai import AzureOpenAI
from db_manager import DatabaseManager
from config import get_cost_rates

logger = logging.getLogger()

# LLM System Messages
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
    
    Args:
        input_text (str): The raw user query to enhance
        
    Returns:
        str: The enhanced query
    """
    # Prepare Azure OpenAI client
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
    
    return answer


def llm_helpee_2xl(input_text: str) -> str:
    """
    Sends PROMPT_ENHANCER_SYSTEM_MESSAGE_2XL to the Azure OpenAI model, logs usage into helpee_logs, and returns the AI output.
    
    Args:
        input_text (str): The raw user query to enhance with detailed prompt engineering
        
    Returns:
        str: The enhanced, detailed prompt
    """
    # Prepare Azure OpenAI client
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

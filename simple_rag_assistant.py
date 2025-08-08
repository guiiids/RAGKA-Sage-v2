"""
Simple Stateless RAG Assistant
- Single-turn responses only
- No conversation memory or session tracking
- No citation persistence 
- Basic knowledge base search and response generation
"""

import os
from typing import List, Dict, Any, Optional
import re
from openai_service import OpenAIService
from cosmos_memory import load_context
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from config import (
    AZURE_OPENAI_ENDPOINT as OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY as OPENAI_KEY,
    AZURE_OPENAI_API_VERSION as OPENAI_API_VERSION,
    CHAT_DEPLOYMENT_GPT4o as CHAT_DEPLOYMENT,
    EMBEDDING_DEPLOYMENT,
    AZURE_SEARCH_SERVICE as SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX as SEARCH_INDEX,
    AZURE_SEARCH_KEY as SEARCH_KEY,
    VECTOR_FIELD,
)


def chunk_document(text: str, max_chunk_size: int = 1000) -> List[str]:
    if len(text) <= max_chunk_size:
        return [text]
    sections = re.split(
        r"((?:^|\n)(?:#+\s+[^\n]+|\d+\.\s+[^\n]+|[A-Z][^\n:]{5,40}:))",
        text,
        flags=re.MULTILINE,
    )
    chunks = []
    current_chunk = ""
    current_headers = []
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        if re.match(
            r"(?:^|\n)(?:#+\s+[^\n]+|\d+\.\s+[^\n]+|[A-Z][^\n:]{5,40}:)",
            section,
            flags=re.MULTILINE,
        ):
            current_headers.append(section.strip())
        elif i > 0:
            if len(current_chunk) + len(section) > max_chunk_size:
                full_chunk = " ".join(current_headers) + " " + current_chunk
                chunks.append(full_chunk)
                current_chunk = section
            else:
                current_chunk += section
    if current_chunk:
        full_chunk = " ".join(current_headers) + " " + current_chunk
        chunks.append(full_chunk)
    if not chunks:
        for i in range(0, len(text), max_chunk_size):
            chunks.append(text[i : i + max_chunk_size])
    return chunks


def extract_metadata(chunk: str) -> Dict[str, Any]:
    metadata = {}
    metadata["is_procedural"] = bool(re.search(r"\d+\.\s+", chunk))
    if re.search(r"^#+\s+", chunk):
        heading_match = re.search(r"^(#+)\s+", chunk)
        metadata["section_level"] = len(heading_match.group(1)) if heading_match else 0
    step_numbers = re.findall(r"(\d+)\.\s+", chunk)
    if step_numbers:
        metadata["steps"] = [int(num) for num in step_numbers]
        metadata["first_step"] = min(metadata["steps"])
        metadata["last_step"] = max(metadata["steps"])
    metadata["is_procedure_start"] = bool(
        re.search(r"(?:how to|steps to|procedure for|guide to)", chunk.lower())
        and metadata.get("is_procedural", False)
    )
    return metadata


def retrieve_with_hierarchy(results: List[Dict]) -> List[Dict]:
    parent_docs = {}
    for result in results:
        parent_id = result.get("parent_id", "")
        if parent_id and parent_id not in parent_docs:
            parent_docs[parent_id] = result.get("relevance", 0.0)
    ordered_results = []
    for parent_id, score in sorted(
        parent_docs.items(), key=lambda x: x[1], reverse=True
    )[:3]:
        parent_chunks = [r for r in results if r.get("parent_id", "") == parent_id]
        for chunk in parent_chunks:
            chunk["metadata"] = extract_metadata(chunk.get("chunk", ""))
        ordered_results.extend(parent_chunks)
    if not ordered_results:
        return results
    return ordered_results


def prioritize_procedural_content(results: List[Dict]) -> List[Dict]:
    for result in results:
        if "metadata" not in result:
            result["metadata"] = extract_metadata(result.get("chunk", ""))
    procedural_results = []
    informational_results = []
    for result in results:
        if result.get("metadata", {}).get("is_procedural", False):
            procedural_results.append(result)
        else:
            informational_results.append(result)
    procedural_results.sort(key=lambda x: x.get("metadata", {}).get("first_step", 999))
    return procedural_results + informational_results


def format_context_text(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    formatted = "\n\n".join(sentence for sentence in sentences if sentence)
    formatted = re.sub(r"(?<=\n\n)([A-Z][^\n:]{5,40})(?=\n\n)", r"**\1**", formatted)
    formatted = re.sub(r"(\d+\.\s+)", r"\n\1", formatted)
    return formatted


def format_procedural_context(text: str) -> str:
    text = re.sub(r"(\d+\.\s+)", r"\n\1", text)
    text = re.sub(r"(\•\s+)", r"\n\1", text)
    text = re.sub(r"([A-Z][^\n:]{5,40}:)", r"\n**\1**\n", text)
    paragraphs = text.split("\n\n")
    formatted = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    return formatted


def is_procedural_content(text: str) -> bool:
    if re.search(r"\d+\.\s+[A-Z]", text):
        return True
    instructional_keywords = ["follow", "steps", "procedure", "instructions", "guide"]
    if any(keyword in text.lower() for keyword in instructional_keywords):
        return True
    return False


# System Prompts
DEFAULT_SYSTEM_PROMPT = """
You are a helpful RAG assistant for enterprise technical support.
Always ground your answers in the provided knowledge base context and include inline citations in the format [1], [2], etc. whenever you reference information from the context or sources.

Guidelines:
- Every factual detail from the KB/context must be cited with an inline [n] marker, referencing the source order as given.
- Do not make statements or claims that cannot be backed by the cited sources.
- Include citations [n] whenever you reference information from the KB context.
- If you don't know the answer, state so clearly.
- Respond in the same language as the user query.
- If the context is unreadable or malformed, notify the user and stop.

Example:
"Product X improves workflow efficiency by 15% [1]. The recommended setup is as follows [2]: ..."

If you cannot find the answer in the context, say "No source found."

<context>
{{CONTEXT}}
</context>
<user_query>
{{QUERY}}
</user_query>
"""

PROCEDURAL_SYSTEM_PROMPT = """
You are a helpful RAG assistant for enterprise procedural support.
Always structure procedures as clear, numbered steps. Ground each instruction in the provided context and include [1], [2], etc. citation markers at the end of any step (or bullet) that is based on the KB/context.

Guidelines:
- Steps should be in logical order with all details needed for accuracy.
- Citations [n] should map to the order of the knowledge base entries as provided.
- Every important step or claim must include a citation as [n] after the step.
- Maintain section headers for clarity if present in the source.
- If you don't know the answer, state so.

Example procedure:
1. Open the main interface [1].
2. Click "Start Configuration" [1].
3. Enter required values as described [2].

<context>
{{CONTEXT}}
</context>
<user_query>
{{QUERY}}
</user_query>
"""


class SimpleRAGAssistant:
    """
    Stateless RAG Assistant for single-turn conversations.
    No memory, no session tracking, no citation persistence.
    """

    def __init__(self, deployment_name: str = CHAT_DEPLOYMENT):
        self.deployment_name = deployment_name
        self.openai_svc = OpenAIService(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_KEY,
            api_version=OPENAI_API_VERSION,
            deployment_name=deployment_name,
        )
        
        from openai import AzureOpenAI
        self.embeddings_client = AzureOpenAI(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_KEY,
            api_version=OPENAI_API_VERSION,
        )
        
        self.search_client = SearchClient(
            endpoint=f"https://{SEARCH_ENDPOINT}.search.windows.net",
            index_name=SEARCH_INDEX,
            credential=AzureKeyCredential(SEARCH_KEY),
        )
        
        # Configurable parameters
        self.temperature = 0.7
        self.max_completion_tokens = 900
        self.top_p = 1.0

    def _make_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for the given text."""
        try:
            resp = self.embeddings_client.embeddings.create(
                model=EMBEDDING_DEPLOYMENT,
                input=text.strip(),
            )
            embedding = resp.data[0].embedding
            return embedding
        except Exception:
            return None

    def _search_kb(self, query: str) -> List[Dict]:
        """Search the knowledge base for relevant chunks."""
        q_vec = self._make_embedding(query)
        if not q_vec:
            return []
        
        vec_q = self.search_client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(
                    vector=q_vec, k_nearest_neighbors=8, fields=VECTOR_FIELD
                )
            ],
            select=["chunk", "title", "parent_id"],
            top=8,
        )
        
        results = [
            {
                "chunk": r.get("chunk", ""),
                "title": r.get("title", "Untitled"),
                "parent_id": r.get("parent_id", ""),
                "relevance": 1.0,
            }
            for r in list(vec_q)
        ]
        
        # Organize/prioritize procedural content
        ordered = retrieve_with_hierarchy(results)
        prioritized = prioritize_procedural_content(ordered)
        return prioritized[:5]

    def _compile_kb_context_sections(self, kb_chunks: List[Dict]) -> str:
        """Format context from knowledge base chunks."""
        entries = []
        for r in kb_chunks:
            chunk = r["chunk"].strip()
            if is_procedural_content(chunk):
                formatted = format_procedural_context(chunk)
            else:
                formatted = format_context_text(chunk)
            entries.append(formatted)
        return "\n\n".join(entries)

    def _select_system_prompt(self, kb_chunks: List[Dict], user_query: str) -> str:
        """Select appropriate system prompt based on content type."""
        if any(
            is_procedural_content(r["chunk"]) for r in kb_chunks
        ) or is_procedural_content(user_query):
            return PROCEDURAL_SYSTEM_PROMPT
        return DEFAULT_SYSTEM_PROMPT

    def _clean_citations(self, answer: str) -> str:
        """Clean up malformed citations and ensure they are in simple [n] format."""
        result = answer

        # Pre-clean any malformed citations or LLM hallucinations
        result = re.sub(r"\[(\d+)\]\s*(?:ref=|href=)[^\]\s\.,;:!?]*", r"[\1]", result)
        result = re.sub(r"\[(\d+)\s+[^]]*\]", r"[\1]", result)
        
        # Remove any existing HTML citation links and convert back to simple [n]
        result = re.sub(r'<a[^>]*class="[^"]*session-citation-link[^"]*"[^>]*>\[(\d+)\]</a>', r'[\1]', result)
        
        # Clean up any other HTML tags that might have been inserted
        result = re.sub(r'<a[^>]*data-citation-id="[^"]*"[^>]*>\[(\d+)\]</a>', r'[\1]', result)

        return result

    def generate_response(self, user_query: str) -> tuple[str, List[Dict]]:
        """
        Generate a single-turn response to the user query.
        Returns: (answer, citations)
        """
        # Search the KB
        kb_chunks = self._search_kb(user_query)
        
        # Compile context
        kb_section = self._compile_kb_context_sections(kb_chunks)
        sys_prompt = self._select_system_prompt(kb_chunks, user_query)
        
        context = ""
        if kb_section:
            context += f"### Knowledge Base Results:\n{kb_section}\n\n"

        # Generate response
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": context + f"\n\nUser question: {user_query}"},
        ]
        
        answer = self.openai_svc.get_chat_response(
            messages=messages, 
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )

        # Create simple citations (no persistence)
        citations = []
        for idx, chunk in enumerate(kb_chunks, 1):
            title = chunk.get("title") or f"Source {idx}"
            citations.append(
                {
                    "index": idx,
                    "display_id": str(idx),
                    "title": title,
                    "content": chunk.get("chunk", ""),
                    "parent_id": chunk.get("parent_id", ""),
                    "id": f"source_{idx}",
                }
            )

        # Clean citations for consistent formatting
        cleaned_answer = self._clean_citations(answer)
        
        return cleaned_answer, citations

    def stream_rag_response(self, user_query: str):
        """
        Stream a response to the user query.
        Yields text chunks and then metadata.
        """
        # Search the KB
        kb_chunks = self._search_kb(user_query)
        
        # Compile context
        kb_section = self._compile_kb_context_sections(kb_chunks)
        sys_prompt = self._select_system_prompt(kb_chunks, user_query)
        
        context = ""
        if kb_section:
            context += f"### Knowledge Base Results:\n{kb_section}\n\n"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": context + f"\n\nUser question: {user_query}"},
        ]

        # Create citations before streaming
        citations = []
        for idx, chunk in enumerate(kb_chunks, 1):
            title = chunk.get("title") or f"Source {idx}"
            citations.append(
                {
                    "index": idx,
                    "display_id": str(idx),
                    "title": title,
                    "content": chunk.get("chunk", ""),
                    "parent_id": chunk.get("parent_id", ""),
                    "id": f"source_{idx}",
                }
            )

        # Emit metadata first
        yield {"sources": citations}

        # Stream the response
        full_answer = ""
        last_yielded = 0
        for chunk in self.openai_svc.get_chat_response_stream(
            messages=messages, 
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        ):
            full_answer += chunk
            # Clean citations in accumulated answer
            cleaned_answer = self._clean_citations(full_answer)
            # Yield only the new content
            new_content = cleaned_answer[last_yielded:]
            if new_content:
                yield new_content
                last_yielded = len(cleaned_answer)


class RAGAssistant(SimpleRAGAssistant):
    """RAG assistant with session-scoped conversational memory."""

    def __init__(self, session_id: str, deployment_name: str = CHAT_DEPLOYMENT):
        super().__init__(deployment_name=deployment_name)
        self.session_id = session_id

    def stream_rag_response(self, user_query: str):
        """Stream response while prepending session history."""
        history = load_context(self.session_id)

        kb_chunks = self._search_kb(user_query)
        kb_section = self._compile_kb_context_sections(kb_chunks)
        sys_prompt = self._select_system_prompt(kb_chunks, user_query)

        context = ""
        if kb_section:
            context += f"### Knowledge Base Results:\n{kb_section}\n\n"

        messages = [{"role": "system", "content": sys_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": context + f"\n\nUser question: {user_query}"})

        citations = []
        for idx, chunk in enumerate(kb_chunks, 1):
            title = chunk.get("title") or f"Source {idx}"
            citations.append(
                {
                    "index": idx,
                    "display_id": str(idx),
                    "title": title,
                    "content": chunk.get("chunk", ""),
                    "parent_id": chunk.get("parent_id", ""),
                    "id": f"source_{idx}",
                }
            )

        yield {"sources": citations}

        full_answer = ""
        last_yielded = 0
        for chunk in self.openai_svc.get_chat_response_stream(
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        ):
            full_answer += chunk
            cleaned_answer = self._clean_citations(full_answer)
            new_content = cleaned_answer[last_yielded:]
            if new_content:
                yield new_content
                last_yielded = len(cleaned_answer)

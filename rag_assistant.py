"""
  Simple Redis-Backed RAG Assistant (No Intelligence)
  - Always: retrieves last N turns from Redis
  - Always: searches knowledge base
  - Combines history + KB context with basic formatting
  - Responds, stores Q&A in Redis
  - No query classification, no conversation intelligence, no routing
  """

import os
from typing import List, Tuple, Dict, Any, Optional, Union
from services.session_memory import (
    SessionMemory,
    PostgresSessionMemory,
    RedisSessionMemory,
)
import re
from openai_service import OpenAIService
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
import json
import time
import hashlib

from services.postgres_citation_service import postgres_citation_service

# --------- Advanced RAG Logic borrowed & adapted from rag_assistant_v2.py --------- #


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


def generate_unique_source_id(content: str = "", timestamp: float = None) -> str:
    if timestamp is None:
        timestamp = int(time.time() * 1000)
    hash_input = f"{content}_{timestamp}".encode("utf-8")
    content_hash = hashlib.md5(hash_input).hexdigest()[:8]
    unique_id = f"S_{timestamp}_{content_hash}"
    return unique_id


# --------- System Prompts --------- #
DEFAULT_SYSTEM_PROMPT = """
You are a helpful RAG assistant for enterprise technical support.
Always ground your answers in the provided knowledge base context and include inline citations in the format [1], [2], etc. whenever you reference information from the context or sources.

Guidelines:
- Every factual detail from the KB/context must be cited with an inline [n] marker, referencing the source order as given.
- Do not make statements or claims that cannot be backed by the cited sources.
- Include only new citations [n] in every response, including all follow-up questions, as long as the information is grounded in the KB context.
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
- When the user asks for more information on any [n], directly (e.g.: "What is step 3?"), or indirectly (e.g.:"Please elaborate on 1.";"Tell me more about 6"), you will alwy assume the user sking about the  provide the full context of that step, including all citations.

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


class EnhancedSimpleRedisRAGAssistant:
    def __init__(
        self,
        session_id: str,
        max_history: int = 5,
        memory: Optional[SessionMemory] = None,
    ):
        self.session_id = session_id
        self.max_history = max_history
        self.memory = memory or PostgresSessionMemory(max_turns=max_history)
        # Using PostgreSQL citation service instead of Redis
        self.openai_svc = OpenAIService(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_KEY,
            api_version=OPENAI_API_VERSION,
            deployment_name=CHAT_DEPLOYMENT,
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

    def _make_embedding(self, text: str) -> Optional[List[float]]:
        try:
            resp = self.embeddings_client.embeddings.create(
                model=EMBEDDING_DEPLOYMENT,
                input=text.strip(),
            )
            embedding = resp.data[0].embedding
            print(f"EMBEDDING DEBUG: type={type(embedding)}, len={len(embedding)}")
            print(f"EMBEDDING DEBUG: resp.data has {len(resp.data)} item(s)")
            if len(resp.data) != 1:
                print(
                    "EMBEDDING DEBUG WARNING: resp.data contains multiple embeddings! This may cause dimensionality errors."
                )
            return embedding
        except Exception:
            return None

    def _search_kb(self, query: str) -> List[Dict]:
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
        # Organize/prioritize procedural content for context window efficiency
        ordered = retrieve_with_hierarchy(results)
        prioritized = prioritize_procedural_content(ordered)
        return prioritized[:5]

    def _compile_history_context(self, history: List[Tuple[str, str]]) -> str:
        turns = []
        for user, assistant in history:
            if user:
                turns.append(f"**User:** {user}")
            if assistant:
                turns.append(f"**Assistant:** {assistant}")
        return "\n\n".join(turns)

    def _compile_kb_context_sections(self, kb_chunks: List[Dict]) -> str:
        # Format context with procedural/section awareness (advanced, taken from v2)
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
        # Simple procedural detection: use procedural prompt if query or chunk suggests
        if any(
            is_procedural_content(r["chunk"]) for r in kb_chunks
        ) or is_procedural_content(user_query):
            return PROCEDURAL_SYSTEM_PROMPT
        return DEFAULT_SYSTEM_PROMPT

    def _clean_citations(self, answer: str) -> str:
        """
        Clean up malformed citations and ensure they are in simple [n] format
        for frontend processing. The frontend will handle the HTML conversion.
        """
        result = answer

        # Pre-clean any malformed citations or LLM hallucinations
        result = re.sub(r"\[(\d+)\]\s*(?:ref=|href=)[^\]\s\.,;:!?]*", r"[\1]", result)
        result = re.sub(r"\[(\d+)\s+[^]]*\]", r"[\1]", result)
        
        # Remove any existing HTML citation links and convert back to simple [n]
        result = re.sub(r'<a[^>]*class="[^"]*session-citation-link[^"]*"[^>]*>\[(\d+)\]</a>', r'[\1]', result)
        
        # Clean up any other HTML tags that might have been inserted
        result = re.sub(r'<a[^>]*data-citation-id="[^"]*"[^>]*>\[(\d+)\]</a>', r'[\1]', result)

        return result

    def _filter_citations(self, answer: str, citations: List[Dict]) -> Tuple[str, List[Dict]]:
        """Return answer and sources filtered to those cited in the answer."""
        if not citations:
            return answer, []

        pattern = re.compile(r"\[(\d+)\]")
        seen: List[int] = []
        for m in pattern.findall(answer):
            try:
                idx = int(m)
            except ValueError:
                continue
            if 1 <= idx <= len(citations) and idx not in seen:
                seen.append(idx)

        if not seen:
            return answer, []

        mapping = {old: new for new, old in enumerate(seen, 1)}

        def repl(match: re.Match) -> str:
            old = int(match.group(1))
            return f"[{mapping.get(old, '')}]" if old in mapping else ""

        filtered_answer = pattern.sub(repl, answer)
        filtered_sources: List[Dict] = []
        for old in seen:
            src = citations[old - 1].copy()
            new_idx = mapping[old]
            src.update({"index": new_idx, "display_id": str(new_idx)})
            filtered_sources.append(src)

        return filtered_answer, filtered_sources

    def _rebuild_citation_map(self, cited_sources):
        """
        Maintain a cumulative map of all sources ever shown/cited in this session,
        so the frontend can resolve citation hyperlinks from any previous message.
        """
        if not hasattr(self, "_display_ordered_citation_map"):
            self._display_ordered_citation_map = {}
        for source in cited_sources:
            uid = source.get("id")
            if uid and uid not in self._display_ordered_citation_map:
                self._display_ordered_citation_map[uid] = source

    def generate_response(self, user_query: str) -> Tuple[str, list]:
        """
        Returns: (html_answer, citations)
        'citations' is a list of dicts matching the [1..N] order used by the system prompt,
        suitable for sidebar or downstream application.
        """
        # 1. Retrieve history from Redis
        history = self.memory.get_history(
            self.session_id, last_n_turns=self.max_history
        )

        print(f"[DEBUG] User query: {user_query}")

        # 2. Search the KB
        kb_chunks = self._search_kb(user_query)
        print(f"[DEBUG] KB Chunks Retrieved: {len(kb_chunks)}")
        for idx, chunk in enumerate(kb_chunks, 1):
            print(
                f"[DEBUG] KB Chunk {idx}: title={chunk.get('title')}, parent_id={chunk.get('parent_id')}, content_snippet={chunk.get('chunk','')[:80]}"
            )

        # 3. Compile the context string (history + KB in advanced format)
        history_section = self._compile_history_context(history)
        kb_section = self._compile_kb_context_sections(kb_chunks)
        sys_prompt = self._select_system_prompt(kb_chunks, user_query)
        context = ""
        if history_section:
            context += f"### Previous Conversation:\n{history_section}\n\n"
        if kb_section:
            context += f"### New Search Results:\n{kb_section}\n\n"

        # 4. Send to LLM (OpenAIService)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": context + f"\n\nUser question: {user_query}"},
        ]
        answer = self.openai_svc.get_chat_response(
            messages=messages, max_completion_tokens=900
        )
        print(f"[DEBUG] LLM Answer: {answer[:500]}")

        # -- Citation assembly: Each kb_chunk corresponds to a [n] marker --
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
        print(f"[DEBUG] Citations Assembled: {len(citations)}")
        for c in citations:
            print(f"[DEBUG] Citation: {c}")

        # --- Clean and filter citations based on answer content ---
        cleaned_answer = self._clean_citations(answer)
        filtered_answer, filtered_citations = self._filter_citations(
            cleaned_answer, citations
        )
        print(
            f"[DEBUG] Answer after filtering citations: {filtered_answer[:500]}"
        )

        # -- Register only the filtered sources with PostgreSQL citation service --
        registered_sources = postgres_citation_service.register_sources(
            self.session_id, filtered_citations
        )
        print(f"[DEBUG] Registered Sources: {registered_sources}")

        # Ensure returned sources have sequential display IDs matching the answer
        for i, source in enumerate(registered_sources, 1):
            source["index"] = i
            source["display_id"] = str(i)

        # Store the turn in Redis
        summary_text = f"User: {user_query}\nAssistant: {filtered_answer}"
        summary = self.openai_svc.summarize_text(summary_text)
        self.memory.store_turn(self.session_id, user_query, filtered_answer, summary)

        # Return the filtered answer and the registered sources
        return filtered_answer, registered_sources

    def stream_rag_response(self, user_query: str):
        """
        Stream partial answer content as it is generated by the LLM.
        After streaming, stores the completed answer in Redis.
        Ensures that all streamed chunks contain citation links (never raw [n]) after citation registration.
        """
        # 1. Retrieve history from Redis
        history = self.memory.get_history(
            self.session_id, last_n_turns=self.max_history
        )

        # 2. Search the KB
        kb_chunks = self._search_kb(user_query)

        # 3. Compile the context string (history + KB in advanced format)
        history_section = self._compile_history_context(history)
        kb_section = self._compile_kb_context_sections(kb_chunks)
        sys_prompt = self._select_system_prompt(kb_chunks, user_query)
        context = ""
        if history_section:
            context += f"### Previous Conversation:\n{history_section}\n\n"
        if kb_section:
            context += f"### New Search Results:\n{kb_section}\n\n"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": context + f"\n\nUser question: {user_query}"},
        ]

        # 4a. Prepare citation metadata
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

        # 4b. Get full answer from LLM
        answer = self.openai_svc.get_chat_response(
            messages=messages, max_completion_tokens=900
        )

        cleaned_answer = self._clean_citations(answer)
        final_answer, filtered_citations = self._filter_citations(
            cleaned_answer, citations
        )

        registered_sources = postgres_citation_service.register_sources(
            self.session_id, filtered_citations
        )
        for i, source in enumerate(registered_sources, 1):
            source["index"] = i
            source["display_id"] = str(i)

        # Emit sources then answer
        yield {"sources": registered_sources}
        yield final_answer

        summary = self.openai_svc.summarize_text(
            f"User: {user_query}\nAssistant: {final_answer}"
        )
        self.memory.store_turn(self.session_id, user_query, final_answer, summary)

    def clear_conversation_history(self) -> None:
        """Clear conversation history for this session"""
        self.memory.clear(self.session_id)
        # Also clear the citation map
        if hasattr(self, "_display_ordered_citation_map"):
            self._display_ordered_citation_map = {}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get basic cache statistics"""
        history = self.memory.get_history(
            self.session_id, last_n_turns=100
        )  # Get all history
        stats = {
            "session_id": self.session_id,
            "conversation_turns": len(history),
            "citation_map_size": len(
                getattr(self, "_display_ordered_citation_map", {})
            ),
        }
        stats.update(self.memory.get_stats())
        return stats

    def clear_cache(self, cache_type: str = None) -> bool:
        """Clear cache (same as clear conversation history for this implementation)"""
        try:
            self.clear_conversation_history()
            return True
        except Exception:
            return False

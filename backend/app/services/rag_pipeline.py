import os
import sys
import logging
from typing import List, Dict, Any, TypedDict
from datetime import datetime
import httpx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from app.core.config import settings
from app.core.database import db_manager
from app.services.embedding import embedding_service

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """
    Typed state representation for the LangGraph workflow.
    """
    query: str
    documents: List[Dict[str, Any]]
    filtered_documents: List[Dict[str, Any]]
    relevance_scores: List[float]
    web_fallback: bool
    answer: str

async def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves matching document chunks from the MongoDB vector index.
    Falls back to a keyword-based regex search if the vector index is offline.
    """
    query = state["query"]
    logger.info(f"[CRAG Node: Retrieve] Executing search query: '{query}'")
    
    raw_vector = embedding_service.embed_query(query)
    query_vector = raw_vector.tolist() if hasattr(raw_vector, "tolist") else raw_vector
    
    documents = []
    if db_manager.db is not None:
        try:
            logger.info("[CRAG Node: Retrieve] Executing vector aggregate search query...")
            cursor = db_manager.client["docutrust_db"]["document_chunks"].aggregate([
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": 100,
                        "limit": 5
                    }
                }
            ])
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                doc.pop("embedding", None)
                documents.append(doc)
            logger.info(f"[CRAG Node: Retrieve] Vector search returned {len(documents)} results.")
        except Exception as e:
            logger.warning(f"[CRAG Node: Retrieve] Vector aggregate query failed: {e}. Falling back to text query.")
            try:
                keywords = [w for w in query.split() if len(w) > 3]
                regex_query = "|".join(keywords) if keywords else query
                cursor = db_manager.db["document_chunks"].find(
                    {"content": {"$regex": regex_query, "$options": "i"}}
                ).limit(5)
                async for doc in cursor:
                    doc["_id"] = str(doc["_id"])
                    doc.pop("embedding", None)
                    documents.append(doc)
                logger.info(f"[CRAG Node: Retrieve] Fallback text query returned {len(documents)} results.")
            except Exception as inner_err:
                logger.error(f"[CRAG Node: Retrieve] Fallback text search failed: {inner_err}")
    else:
        logger.error("[CRAG Node: Retrieve] MongoDB reference is not initialized.")

    if documents:
        try:
            doc_ids = list({doc.get("metadata", {}).get("source_doc_id") for doc in documents if doc.get("metadata", {}).get("source_doc_id")})
            if doc_ids:
                db_ref = db_manager.db if db_manager.db is not None else db_manager.client["docutrust_db"]
                await db_ref["documents"].update_many(
                    {"_id": {"$in": doc_ids}},
                    {"$set": {"last_accessed_at": datetime.utcnow()}}
                )
                logger.info(f"[CRAG Node: Retrieve] Updated last_accessed_at for: {doc_ids}")
        except Exception as update_err:
            logger.error(f"[CRAG Node: Retrieve] Failed to update document access timestamps: {update_err}")

    return {"query": query, "documents": documents}

async def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grades retrieved chunks using a cross-encoder model.
    Triggers the web fallback routing if all chunks fail the relevance check.
    """
    query = state["query"]
    documents = state["documents"]
    
    if not documents:
        logger.info("[CRAG Node: Grade] Empty document state. Routing to web search.")
        return {"filtered_documents": [], "relevance_scores": [], "web_fallback": True}
        
    logger.info(f"[CRAG Node: Grade] Evaluating {len(documents)} retrieved passages...")
    doc_contents = [doc["content"] for doc in documents]
    scores = embedding_service.compute_relevance_scores(query, doc_contents)
    
    filtered_docs = []
    relevance_scores = []
    RELEVANCE_THRESHOLD = 0.4
    
    for doc, score in zip(documents, scores):
        logger.info(f"[CRAG Node: Grade] Score: {score:.4f} for: {doc.get('metadata', {}).get('filename')}")
        if score >= RELEVANCE_THRESHOLD:
            filtered_docs.append(doc)
            relevance_scores.append(score)
            
    web_fallback = False
    if not filtered_docs:
        logger.info("[CRAG Node: Grade] No relevant passages identified. Activating web search.")
        web_fallback = True
    else:
        logger.info(f"[CRAG Node: Grade] Retained {len(filtered_docs)} relevant passages.")
        
    return {
        "filtered_documents": filtered_docs,
        "relevance_scores": relevance_scores,
        "web_fallback": web_fallback
    }

async def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Executes search query optimization and fetches fallback web results.
    """
    query = state["query"]
    filtered_documents = state["filtered_documents"]
    
    logger.info(f"[CRAG Node: WebSearch] Re-formulating search query: '{query}'")
    words = [w for w in query.split() if w.lower() not in {"please", "find", "search", "what", "is", "explain", "tell"}]
    search_query = " ".join(words[:6]) if words else query
    
    web_results = []
    tavily_key = settings.TAVILY_API_KEY
    
    if tavily_key:
        logger.info(f"[CRAG Node: WebSearch] Executing Tavily query: '{search_query}'")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": search_query,
                        "search_depth": "basic",
                        "max_results": 3
                    },
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    for idx, res in enumerate(data.get("results", [])):
                        web_results.append({
                            "content": res.get("content", ""),
                            "metadata": {
                                "source_doc_id": "web_fallback",
                                "filename": res.get("url", "Web Search"),
                                "chunk_index": idx,
                                "title": res.get("title", "Web Result")
                            }
                        })
                    logger.info(f"[CRAG Node: WebSearch] Retrieved {len(web_results)} results from Tavily.")
        except Exception as e:
            logger.error(f"[CRAG Node: WebSearch] Tavily API request failed: {e}")
            
    if not web_results:
        logger.info(f"[CRAG Node: WebSearch] Querying DuckDuckGo fallback scraper: '{search_query}'")
        try:
            async with httpx.AsyncClient() as client:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                response = await client.get(
                    f"https://html.duckduckgo.com/html/?q={search_query}",
                    headers=headers,
                    timeout=8.0
                )
                if response.status_code == 200:
                    if BeautifulSoup:
                        soup = BeautifulSoup(response.text, "html.parser")
                        links = soup.find_all("a", class_="result__snippet")
                        for idx, link in enumerate(links[:3]):
                            web_results.append({
                                "content": link.get_text().strip(),
                                "metadata": {
                                    "source_doc_id": "web_fallback",
                                    "filename": "DuckDuckGo Web Search",
                                    "chunk_index": idx
                                }
                            })
                    else:
                        parts = response.text.split('<a class="result__snippet"')
                        for idx, part in enumerate(parts[1:4]):
                            sub = part.split('</a>')[0]
                            snippet = sub.split('>')[-1].strip()
                            if snippet:
                                web_results.append({
                                    "content": snippet,
                                    "metadata": {
                                        "source_doc_id": "web_fallback",
                                        "filename": "DuckDuckGo Web Search (Raw)",
                                        "chunk_index": idx
                                    }
                                })
                    logger.info(f"[CRAG Node: WebSearch] DuckDuckGo search found {len(web_results)} results.")
        except Exception as e:
            logger.error(f"[CRAG Node: WebSearch] DuckDuckGo scraping failed: {e}")
            
    combined_docs = list(filtered_documents) + web_results
    return {"filtered_documents": combined_docs}

async def generate(state: GraphState) -> Dict[str, Any]:
    """
    Synthesizes the response using context passages.
    Attempts local Flan-T5 generation, falling back to RoBERTa QA extraction, and finally extractive summary.
    """
    query = state["query"]
    documents = state["filtered_documents"]
    
    if not documents:
        return {"answer": "No reference materials found in document chunks or web fallbacks."}
        
    logger.info(f"[CRAG Node: Generate] Compiling final answer using {len(documents)} passages.")
    context_blocks = []
    citations = []
    
    for doc in documents:
        meta = doc.get("metadata", {})
        source_name = meta.get("filename", "Corporate Policy")
        chunk_idx = meta.get("chunk_index", 0)
        content = doc.get("content", "")
        
        cite_tag = f"[{source_name} (Chunk {chunk_idx})]"
        citations.append(cite_tag)
        context_blocks.append(content)
        
    context = "\n\n".join(context_blocks)

    # Tier 1: Try Text Generation with Flan-T5 (Synthesized Natural Answer)
    try:
        from transformers import pipeline
        logger.info("[CRAG Node: Generate] Attempting text synthesis with google/flan-t5-base...")
        generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
        
        prompt = (
            f"Use the following context to answer the question at the end.\n"
            f"If the context does not contain the answer, say that the information is not available.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        
        result = generator(prompt, max_new_tokens=150, temperature=0.2)
        answer_text = result[0].get("generated_text", "").strip()
        
        if answer_text and len(answer_text) > 3 and not answer_text.lower().startswith("i don") and "not available" not in answer_text.lower():
            logger.info("[CRAG Node: Generate] Flan-T5 text synthesis succeeded.")
            return {"answer": answer_text}
        else:
            raise ValueError("Flan-T5 response empty or low-quality.")
            
    except Exception as e:
        logger.warning(f"[CRAG Node: Generate] Tier 1 Flan-T5 generation failed/skipped: {e}. Trying Tier 2 RoBERTa QA...")
        
        # Tier 2: Try RoBERTa Span Extraction
        try:
            from transformers import pipeline
            logger.info("[CRAG Node: Generate] Attempting span extraction with deepset/roberta-base-squad2...")
            qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
            
            result = qa_pipeline(question=query, context=context)
            answer_text = result.get("answer", "").strip()
            confidence = result.get("score", 0.0)
            
            if confidence > 0.05 and len(answer_text) > 3:
                logger.info("[CRAG Node: Generate] RoBERTa span extraction succeeded.")
                return {"answer": f"According to the source documents, {answer_text}."}
            else:
                raise ValueError("RoBERTa QA score too low.")
                
        except Exception as inner_e:
            logger.warning(f"[CRAG Node: Generate] Tier 2 QA extraction failed/skipped: {inner_e}. Falling back to Tier 3 Extractive Synthesis...")
            
            # Tier 3: Clean Extractive Synthesis (returns text context, leaves citation IDs to metadata)
            query_words = [w.lower() for w in query.split() if len(w) > 3 and w.lower() not in {"what", "where", "when", "how", "why"}]
            summary_sentences = []
            
            for doc in documents:
                content = doc.get("content", "")
                sentences = [s.strip() for s in content.replace("\n", " ").split(". ") if s.strip()]
                
                for sentence in sentences:
                    if any(w in sentence.lower() for w in query_words):
                        summary_sentences.append(sentence)
                        
                if not summary_sentences and sentences:
                    summary_sentences.append(sentences[0])
                    
            unique_sentences = list(dict.fromkeys(summary_sentences))[:3]
            synthesized_text = ". ".join(unique_sentences)
            if not synthesized_text.endswith("."):
                synthesized_text += "."
                
            logger.info("[CRAG Node: Generate] Tier 3 Extractive Synthesis succeeded.")
            return {"answer": synthesized_text}

def decide_to_generate(state: GraphState) -> str:
    """
    Routes the state flow based on the web_fallback indicator.
    """
    if state["web_fallback"]:
        return "web_search"
    return "generate"

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

crag_pipeline = workflow.compile()

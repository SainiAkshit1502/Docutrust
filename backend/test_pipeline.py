import os
import sys
import asyncio
import logging

# Ensure backend root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import connect_to_mongo, close_mongo_connection
from app.services.rag_pipeline import crag_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

async def test_crag():
    print("Initializing DocuTrust CRAG Pipeline Verification Test...")
    try:
        # Establish DB connection
        await connect_to_mongo()
        
        # Test Query
        test_query = "What is the summary of AIML Major Project?"
        print(f"\nRunning query through LangGraph CRAG workflow: '{test_query}'...")
        
        # Invoke LangGraph StateGraph
        initial_state = {
            "query": test_query,
            "documents": [],
            "filtered_documents": [],
            "relevance_scores": [],
            "web_fallback": False,
            "answer": ""
        }
        
        result_state = await crag_pipeline.ainvoke(initial_state)
        
        print("\n=== CRAG PIPELINE RESULT STATE ===")
        print(f"Query: {result_state.get('query')}")
        print(f"Retrieved Document Chunks: {len(result_state.get('documents', []))}")
        print(f"Kept Relevant Chunks: {len(result_state.get('filtered_documents', []))}")
        print(f"Relevance Scores: {result_state.get('relevance_scores')}")
        print(f"Web Fallback Triggered: {result_state.get('web_fallback')}")
        print(f"\nFinal Generated Answer:\n{result_state.get('answer')}")
        print("===================================\n")
        
        assert "answer" in result_state and len(result_state["answer"]) > 0, "RAG pipeline failed to return an answer."
        print("SUCCESS: CRAG Pipeline Verification Test completed successfully!")
        
    except Exception as e:
        print(f"ERROR: CRAG Pipeline Verification Test failed: {e}", file=sys.stderr)
    finally:
        await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(test_crag())

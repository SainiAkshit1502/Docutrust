import os
import sys
import io
import uuid
from datetime import datetime
import logging
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import settings
from app.core.database import connect_to_mongo, close_mongo_connection, db_manager
from app.services.embedding import embedding_service
from app.services.document_processor import document_processor
from app.services.rag_pipeline import crag_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    query: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles connection management lifecycle hooks for FastAPI startup and shutdown.
    """
    try:
        await connect_to_mongo()
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}. Starting in degraded mode.")
    
    logger.info("Pre-warming transformer modules...")
    try:
        embedding_service.initialize_models()
    except Exception as e:
        logger.warning(f"Failed to pre-initialize models: {e}. Models will load on demand.")

    yield
    await close_mongo_connection()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="DocuTrust - Enterprise Advanced RAG Platform with Automated Self-Correction",
    version="1.0.0",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check_root():
    """
    Root status endpoint displaying backend API metadata and database health state.
    """
    db_status = "unconnected"
    db_error = None
    if db_manager.client is not None:
        try:
            await db_manager.client.admin.command('ping')
            db_status = "connected"
        except Exception as e:
            db_status = "disconnected"
            db_error = str(e)
            logger.error(f"MongoDB ping failed: {e}")
            
    return {
        "status": "online",
        "api_version": "1.0.0",
        "database": {
            "status": db_status,
            "error": db_error,
            "database_name": settings.MONGODB_DB_NAME
        },
        "embedding_pipeline": {
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "cross_encoder_model": settings.CROSS_ENCODER_MODEL_NAME
        }
    }

@app.get(f"{settings.API_V1_STR}/health")
async def health_check():
    """
    Performs standard status ping to verify API service availability.
    """
    db_status = "unconnected"
    if db_manager.client is not None:
        try:
            await db_manager.client.admin.command('ping')
            db_status = "healthy"
        except Exception as e:
            logger.error(f"Health check failed for MongoDB connection: {e}")
            db_status = "unhealthy"
    
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "database": db_status
    }

@app.post(f"{settings.API_V1_STR}/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads and processes a PDF file. Verifies deduplication content hash rules, 
    purges oldest records if capacity threshold is exceeded, and embeds/stores vectors.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        db_ref = db_manager.db if db_manager.db is not None else db_manager.client["docutrust_db"]
        if db_ref is None:
            raise HTTPException(status_code=503, detail="Database connection is not available.")

        contents = await file.read()
        file_stream = io.BytesIO(contents)
        file_hash = document_processor.compute_md5(file_stream)
        
        docs_collection = db_ref["documents"]
        existing_doc = await docs_collection.find_one({"file_hash": file_hash})
        if existing_doc:
            raise HTTPException(status_code=400, detail="This document is already indexed.")
            
        doc_count = await docs_collection.count_documents({})
        if doc_count >= 5:
            cursor = docs_collection.find({}).sort("last_accessed_at", 1).limit(1)
            oldest_docs = await cursor.to_list(length=1)
            if oldest_docs:
                oldest_doc = oldest_docs[0]
                oldest_id = oldest_doc["_id"]
                oldest_filename = oldest_doc.get("filename", "Unknown")
                logger.info(f"Capacity cap reached. Evicting oldest accessed document: '{oldest_filename}'")
                
                await docs_collection.delete_one({"_id": oldest_id})
                await db_ref["document_chunks"].delete_many({"metadata.source_doc_id": oldest_id})
                logger.info(f"Cleaned up metadata and chunks for evicted document: '{oldest_filename}'")
        
        document_id = str(uuid.uuid4())
        mongo_payloads = document_processor.process_pdf(
            file_source=file_stream,
            document_id=document_id,
            filename=file.filename,
            additional_metadata={"uploaded_at": datetime.utcnow()}
        )
        
        if not mongo_payloads:
            raise HTTPException(status_code=400, detail="No readable text could be extracted from the PDF.")
            
        chunks_collection = db_ref["document_chunks"]
        await chunks_collection.insert_many(mongo_payloads)
        
        await docs_collection.insert_one({
            "_id": document_id,
            "filename": file.filename,
            "file_hash": file_hash,
            "chunk_count": len(mongo_payloads),
            "status": "processed",
            "created_at": datetime.utcnow(),
            "last_accessed_at": datetime.utcnow()
        })
        
        return {
            "message": "Document uploaded and embedded successfully.",
            "document_id": document_id,
            "filename": file.filename,
            "chunks_vectorized": len(mongo_payloads)
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Upload and processing pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post(f"{settings.API_V1_STR}/chat")
async def chat_interaction(request: ChatRequest):
    """
    Routes conversational search query tasks into the compiled LangGraph workflow.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        initial_state = {
            "query": request.query,
            "documents": [],
            "filtered_documents": [],
            "relevance_scores": [],
            "web_fallback": False,
            "answer": ""
        }
        
        logger.info(f"Invoking RAG workflow for search query: '{request.query}'")
        result_state = await crag_pipeline.ainvoke(initial_state)
        
        citations = []
        scores = result_state.get("relevance_scores", [])
        for idx, doc in enumerate(result_state.get("filtered_documents", [])):
            score = float(scores[idx]) if idx < len(scores) else 1.0
            citations.append({
                "content": doc.get("content", ""),
                "source": doc.get("metadata", {}).get("filename", "Web Context"),
                "chunk_index": doc.get("metadata", {}).get("chunk_index", 0),
                "relevance_score": score
            })

        return {
            "query": request.query,
            "answer": result_state.get("answer", "Failed to compile response."),
            "web_fallback_triggered": result_state.get("web_fallback", False),
            "citations": citations
        }
    except Exception as e:
        logger.error(f"Chat route evaluation failure: {e}")
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting uvicorn server on {settings.HOST}:{settings.PORT}...")
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

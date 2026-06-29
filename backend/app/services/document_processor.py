import os
import sys
import io
import logging
from typing import List, Dict, Any, Union
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles PDF document text extraction, text splitting, 
    vector embedding generation, and database document indexing.
    """
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> None:
        """
        Initializes the document processor with the text splitter configuration.
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

    def compute_md5(self, file_source: Union[str, io.BytesIO]) -> str:
        """
        Computes the MD5 checksum of a file path or in-memory byte stream.
        """
        import hashlib
        hash_md5 = hashlib.md5()
        if isinstance(file_source, str):
            with open(file_source, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            current_pos = file_source.tell()
            file_source.seek(0)
            for chunk in iter(lambda: file_source.read(4096), b""):
                hash_md5.update(chunk)
            file_source.seek(current_pos)
            
        return hash_md5.hexdigest()

    def initialize_embedding_model(self) -> None:
        """
        Performs lazy initialization of the local embedding transformer model.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading transformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load sentence-transformer model: {e}. Defaulting to mock.")
                self._model = "mock"

    def extract_text_from_pdf_path(self, file_path: str) -> str:
        """
        Extracts all textual content from a local PDF file path.
        """
        logger.info(f"Reading PDF from path: {file_path}")
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to read PDF from path: {e}")
            raise e

    def extract_text_from_pdf_stream(self, file_stream: io.BytesIO) -> str:
        """
        Extracts all textual content from a file-like byte stream.
        """
        logger.info("Reading PDF from file-like stream.")
        text = ""
        try:
            reader = PdfReader(file_stream)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to read PDF from stream: {e}")
            raise e

    def split_text(self, text: str) -> List[str]:
        """
        Partitions text into structured chunks based on text splitter settings.
        """
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split document text into {len(chunks)} chunks.")
        return chunks

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Transforms text chunks into dense float array embedding vectors.
        """
        if not chunks:
            return []
            
        self.initialize_embedding_model()

        if self._model == "mock":
            logger.warning("Generating mock vector embeddings.")
            dimension = 384
            embeddings = []
            for text in chunks:
                seed = sum(ord(c) for c in text) % 1000
                import numpy as np
                np.random.seed(seed)
                vector = np.random.randn(dimension)
                norm = np.linalg.norm(vector)
                normalized_vector = (vector / norm).tolist() if norm > 0 else [0.0] * dimension
                embeddings.append(normalized_vector)
            return embeddings

        vectors = self._model.encode(
            chunks, 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        return vectors.tolist()

    def prepare_vector_documents(
        self, 
        chunks: List[str], 
        embeddings: List[List[float]], 
        document_id: str,
        filename: str,
        additional_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepares document chunk dictionary payloads for database ingestion.
        """
        mongo_documents = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = {
                "source_doc_id": document_id,
                "filename": filename,
                "chunk_index": idx,
                "chunk_length": len(chunk),
                "created_at": datetime.utcnow()
            }
            if additional_metadata:
                metadata.update(additional_metadata)

            doc = {
                "content": chunk,
                "embedding": embedding,
                "metadata": metadata
            }
            mongo_documents.append(doc)
        
        return mongo_documents

    def process_pdf(
        self, 
        file_source: Union[str, io.BytesIO], 
        document_id: str,
        filename: str,
        additional_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes text extraction, recursive splitting, and chunk vector embedding generation.
        """
        if isinstance(file_source, str):
            raw_text = self.extract_text_from_pdf_path(file_source)
        else:
            raw_text = self.extract_text_from_pdf_stream(file_source)
            
        if not raw_text.strip():
            logger.warning(f"Extracted empty text from: {filename}")
            return []
            
        chunks = self.split_text(raw_text)
        embeddings = self.generate_embeddings(chunks)
        
        return self.prepare_vector_documents(
            chunks=chunks, 
            embeddings=embeddings, 
            document_id=document_id, 
            filename=filename, 
            additional_metadata=additional_metadata
        )

document_processor = DocumentProcessor(
    model_name=getattr(settings, "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
)

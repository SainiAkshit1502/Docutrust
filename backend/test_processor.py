import os
import sys

# Add parent directory of 'app' to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.document_processor import document_processor

def test_chunking_and_embedding():
    print("Initializing DocumentProcessor Pipeline Self-Test...")
    
    # 1. Test Text Splitting
    print("\n1. Testing Text Splitting...")
    sample_text = (
        "This is the first sentence of our corporate policy regarding data protection. "
        "All customer profiles must be held securely in our MongoDB database. "
        "The retrieval pipeline utilizes local cross-encoder models to check query relevance before displaying text. "
        "This ensures that basic AI document search portals do not suffer from heavy hallucinations."
    )
    
    # Temporarily set splitter chunk sizes to be smaller for testing boundary splits
    original_chunk_size = document_processor.chunk_size
    original_chunk_overlap = document_processor.chunk_overlap
    
    document_processor.text_splitter._chunk_size = 100
    document_processor.text_splitter._chunk_overlap = 20
    
    chunks = document_processor.split_text(sample_text)
    print(f"Chunks generated ({len(chunks)}):")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {repr(c)}")
        
    assert len(chunks) > 0, "No chunks generated from test text."
    
    # Restore original settings
    document_processor.text_splitter._chunk_size = original_chunk_size
    document_processor.text_splitter._chunk_overlap = original_chunk_overlap
    
    # 2. Test Embedding Generation
    print("\n2. Testing Embedding Generation...")
    # Generate embeddings on a list of chunks
    embeddings = document_processor.generate_embeddings(chunks)
    print(f"Successfully generated {len(embeddings)} vectors.")
    print(f"Vector dimensions: {len(embeddings[0])}")
    assert len(embeddings) == len(chunks), "Number of embedding vectors does not match chunk count."
    
    # 3. Test MongoDB Payload Formatting
    print("\n3. Testing MongoDB Payload Formatting...")
    payloads = document_processor.prepare_vector_documents(
        chunks=chunks,
        embeddings=embeddings,
        document_id="test_doc_999",
        filename="corporate_security_policy.pdf",
        additional_metadata={"department": "IT Compliance"}
    )
    
    print("Sample MongoDB Database Payload (First Chunk):")
    first_payload = payloads[0]
    print(f"  Content: {repr(first_payload['content'])}")
    print(f"  Embedding length: {len(first_payload['embedding'])}")
    print(f"  Metadata: {first_payload['metadata']}")
    
    assert "content" in first_payload, "Payload missing 'content' key."
    assert "embedding" in first_payload, "Payload missing 'embedding' key."
    assert first_payload['metadata']['source_doc_id'] == "test_doc_999", "Metadata missing source document ID."
    assert first_payload['metadata']['department'] == "IT Compliance", "Metadata missing additional custom fields."
    
    print("\nSUCCESS: All document processor pipeline checks passed successfully!")

if __name__ == "__main__":
    test_chunking_and_embedding()

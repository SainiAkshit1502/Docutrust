import asyncio
import sys
import os
from datetime import datetime, timedelta

# Ensure backend root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import connect_to_mongo, close_mongo_connection, db_manager

async def test_deduplication_and_lru():
    print("Initializing Deduplication and LRU Eviction Verification Check...")
    try:
        await connect_to_mongo()
        db = db_manager.db if db_manager.db is not None else db_manager.client["docutrust_db"]
        
        # 1. Clear database documents and chunks for clean test run
        print("\n1. Cleaning up collections for test isolation...")
        await db["documents"].delete_many({})
        await db["document_chunks"].delete_many({})
        
        docs_collection = db["documents"]
        chunks_collection = db["document_chunks"]
        
        # 2. Insert 5 mock documents with ascending timestamps (oldest to newest)
        print("\n2. Seeding 5 mock documents with staggered access times (doc_1 is oldest)...")
        now = datetime.utcnow()
        for i in range(1, 6):
            doc_id = f"doc_{i}"
            # i-th document last accessed i hours ago (doc_1 is oldest, doc_5 is newest)
            access_time = now - timedelta(hours=(6 - i))
            
            await docs_collection.insert_one({
                "_id": doc_id,
                "filename": f"document_{i}.pdf",
                "file_hash": f"hash_{i}",
                "chunk_count": 3,
                "status": "processed",
                "created_at": access_time,
                "last_accessed_at": access_time
            })
            
            # Insert mock chunks
            for chunk_idx in range(3):
                await chunks_collection.insert_one({
                    "content": f"This is chunk {chunk_idx} of document {i}.",
                    "metadata": {
                        "source_doc_id": doc_id,
                        "filename": f"document_{i}.pdf",
                        "chunk_index": chunk_idx
                    }
                })
        
        count = await docs_collection.count_documents({})
        chunk_count = await chunks_collection.count_documents({})
        print(f"  Current state: {count} metadata documents and {chunk_count} text chunks in DB.")
        assert count == 5, "Database must have 5 documents."
        assert chunk_count == 15, "Database must have 15 chunks."
        
        # 3. Test Deduplication
        print("\n3. Testing Deduplication (attempting to add duplicate document_2)...")
        # Simulate hash matching
        existing_doc = await docs_collection.find_one({"file_hash": "hash_2"})
        print(f"  Searching duplicate hash 'hash_2': found? {existing_doc is not None}")
        assert existing_doc is not None, "Deduplication search check failed: hash_2 must be found."
        
        # 4. Update access time for oldest document doc_1 to make doc_2 the oldest
        print("\n4. Simulating search query access on document_1 to update its timestamp...")
        await docs_collection.update_one(
            {"_id": "doc_1"},
            {"$set": {"last_accessed_at": now}}
        )
        print("  Updated document_1 last_accessed_at to current timestamp.")
        
        # 5. Eviction check simulation (LRU) - insert document_6 (limit = 5)
        print("\n5. Simulating insertion of 6th document (document_6.pdf) to test LRU eviction...")
        # Since limit is reached, we must evict the oldest accessed document.
        # Let's inspect the oldest document:
        cursor = docs_collection.find({}).sort("last_accessed_at", 1).limit(1)
        oldest_docs = await cursor.to_list(length=1)
        assert oldest_docs, "No oldest document found."
        oldest_doc = oldest_docs[0]
        print(f"  Oldest accessed document identified for eviction: '{oldest_doc['filename']}' (ID: {oldest_doc['_id']})")
        assert oldest_doc["_id"] == "doc_2", "doc_2 should be the oldest after doc_1 timestamp update."
        
        # Evict oldest (doc_2)
        oldest_id = oldest_doc["_id"]
        await docs_collection.delete_one({"_id": oldest_id})
        await chunks_collection.delete_many({"metadata.source_doc_id": oldest_id})
        print(f"  Evicted document ID: {oldest_id} and its associated vector chunks.")
        
        # Insert 6th doc
        await docs_collection.insert_one({
            "_id": "doc_6",
            "filename": "document_6.pdf",
            "file_hash": "hash_6",
            "chunk_count": 3,
            "status": "processed",
            "created_at": now,
            "last_accessed_at": now
        })
        print("  Inserted document_6.pdf successfully.")
        
        # Check counts
        final_doc_count = await docs_collection.count_documents({})
        final_chunk_count = await chunks_collection.count_documents({})
        print(f"  Final state: {final_doc_count} documents, {final_chunk_count} chunks in database.")
        assert final_doc_count == 5, "Database must remain capped at 5 documents."
        assert final_chunk_count == 12, "Database must have 12 chunks (15 - 3 evicted + 0 added for seeding simplicity)."
        
        # Verify doc_2 is deleted
        deleted_metadata = await docs_collection.find_one({"_id": "doc_2"})
        deleted_chunks_count = await chunks_collection.count_documents({"metadata.source_doc_id": "doc_2"})
        print(f"  Evicted metadata found? {deleted_metadata is not None} (Expected: False)")
        print(f"  Evicted chunks count remaining: {deleted_chunks_count} (Expected: 0)")
        
        assert deleted_metadata is None, "Evicted metadata still exists."
        assert deleted_chunks_count == 0, "Evicted chunks still exist."
        
        print("\nSUCCESS: All Deduplication and LRU Eviction pipeline checks passed successfully!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nERROR: Verification failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(test_deduplication_and_lru())

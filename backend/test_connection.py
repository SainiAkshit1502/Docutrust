import asyncio
import sys
import os

# Add the parent directory of 'app' to python path to run script directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings
from app.core.database import connect_to_mongo, close_mongo_connection

async def test_connection():
    print("Initializing DocuTrust DB Connection Test...")
    print(f"Target URL: {settings.MONGODB_URL}")
    print(f"Target DB: {settings.MONGODB_DB_NAME}")
    try:
        await connect_to_mongo()
        print("\nSUCCESS: DocuTrust successfully connected to MongoDB.")
        await close_mongo_connection()
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Connection validation failed: {e}", file=sys.stderr)
        print("\nTroubleshooting Tips:", file=sys.stderr)
        print("1. Ensure MongoDB service is running (e.g. net start MongoDB on Windows).", file=sys.stderr)
        print("2. Check if the MONGODB_URL in backend/.env is correct.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_connection())

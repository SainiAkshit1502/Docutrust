import logging
import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    client: AsyncIOMotorClient = None
    db = None

db_manager = DatabaseManager()

async def connect_to_mongo() -> None:
    """
    Establish asynchronous connection to MongoDB and ping database to verify status.
    """
    logger.info(f"Connecting to MongoDB at {settings.MONGODB_URL}...")
    db_manager.client = AsyncIOMotorClient(settings.MONGODB_URL, tlsCAFile=certifi.where())
    db_manager.db = db_manager.client[settings.MONGODB_DB_NAME]
    try:
        # Perform a ping to verify connection is successful
        await db_manager.client.admin.command('ping')
        logger.info(f"Connected to MongoDB database '{settings.MONGODB_DB_NAME}' successfully!")
    except Exception as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        raise e

async def close_mongo_connection() -> None:
    """
    Gracefully terminate MongoDB client connections.
    """
    logger.info("Closing MongoDB connection...")
    if db_manager.client is not None:
        db_manager.client.close()
        logger.info("MongoDB connection closed successfully.")

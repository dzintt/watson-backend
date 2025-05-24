from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from dotenv import load_dotenv
from uuid import uuid4
import time
import os

load_dotenv()

client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
database = client[os.getenv("DATABASE_NAME")]

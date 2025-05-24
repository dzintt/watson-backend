from fastapi import FastAPI, APIRouter
import os
from routes.conversation_handler import router as conversation_handler_router

from dotenv import load_dotenv

import uvicorn

load_dotenv()


def main():
    app = FastAPI()
    
    api_router = APIRouter(prefix="/api")
    
    api_router.include_router(conversation_handler_router, prefix="/conversation")
    
    @api_router.get("/")
    async def root():
        return {"message": "success"}
    
    app.include_router(api_router)

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

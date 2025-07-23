import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import routers
from supa import (
    auth_router,
    user_router,
    history_router,
    analytics_router,
    files_router,
)
from llm import processing_router, system_router

load_dotenv()


app = FastAPI(
    title="Document Processing API",
    description="API for processing documents with Supabase authentication",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add error handling middleware
@app.middleware("http")
async def error_handling_middleware(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"Unhandled error: {str(e)}")
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(e)},
        )


# Include routers
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(history_router)
app.include_router(analytics_router)
app.include_router(files_router)
app.include_router(processing_router)
app.include_router(system_router)


@app.get("/")
async def read_root():
    return {
        "message": "Document Processing API with Supabase Authentication",
        "version": "1.0.0",
        "endpoints": {
            "authentication": {
                "signup": "POST /auth/signup",
                "signin": "POST /auth/signin",
                "signout": "POST /auth/signout",
                "test": "GET /auth/test/supabase",
            },
            "user": {
                "profile_get": "GET /user/profile",
                "profile_update": "PUT /user/profile",
                "current_user": "GET /user/me",
            },
            "processing": {
                "generate": "POST /generate",
            },
            "history": {
                "list": "GET /history/",
                "get": "GET /history/{record_id}",
                "delete": "DELETE /history/{record_id}",
            },
            "analytics": {
                "full": "GET /analytics/",
                "summary": "GET /analytics/summary",
            },
            "system": {"health": "GET /system/health"},
        },
        "documentation": {"interactive": "/docs", "openapi": "/openapi.json"},
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8001, reload=True)

"""
main.py - FastAPI application entry point
"""
import logging
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import routers
from .routers import (
    auth_routes,
    chat_routes,
    config_routes,
    admin_routes,
    user_routes,
    analytics_routes,
    ingest_routes
)

# Import other modules
from .database import init_database, log_error
from .widget import generate_widget_js
from .config import DEFAULT_TENANT, DEFAULT_AGENT

logging.basicConfig(level=logging.INFO)

# Create FastAPI app
app = FastAPI(
    title="Multi-Tenant RAG Chatbot",
    description="A multi-tenant RAG chatbot with advanced features",
    version="7.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize database
init_database()

# Mount static files if directory exists
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(auth_routes.router)
app.include_router(chat_routes.router)
app.include_router(config_routes.router)
app.include_router(user_routes.router)
app.include_router(admin_routes.router)
app.include_router(analytics_routes.router)
app.include_router(ingest_routes.router)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Log HTTP errors and return JSON response."""
    log_error(
        endpoint=str(request.url.path),
        tenant=request.query_params.get("tenant"),
        agent=request.query_params.get("agent"),
        message=str(exc.detail),
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    log_error(
        endpoint=str(request.url.path),
        tenant=request.query_params.get("tenant"),
        agent=request.query_params.get("agent"),
        message=str(exc),
    )
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# Widget endpoint
@app.get("/widget.js", response_class=PlainTextResponse)
async def get_widget(
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT)
):
    """Get widget JavaScript"""
    return generate_widget_js(tenant, agent)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Tenant RAG Chatbot API",
        "version": "7.0",
        "docs": "/docs",
        "admin": "/admin.html"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

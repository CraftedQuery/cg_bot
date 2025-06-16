"""
RAG Chatbot - Multi-tenant RAG chatbot with advanced features
"""

__version__ = "7.0"
__author__ = "RAG Chatbot Team"

# Importing the FastAPI app here would require all optional dependencies
# (e.g. langchain) to be installed at package import time. To allow importing
# parts of this package without those extras, the app is exposed lazily via
# `get_app()`.

def get_app():
    from .main import app
    return app

__all__ = ["get_app"]

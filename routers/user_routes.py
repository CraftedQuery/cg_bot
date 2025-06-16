"""user_routes.py - User landing page endpoints"""
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["user"])

@router.get("/user", response_class=HTMLResponse)
async def get_user_interface():
    """Redirect to user landing page"""
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Redirecting...</title>
            <script>window.location.href = '/user.html';</script>
        </head>
        <body>
            <p>Redirecting to user interface...</p>
        </body>
        </html>
        """
    )

@router.get("/user.html", response_class=HTMLResponse)
async def serve_user_html():
    """Serve the user landing page HTML"""
    try:
        user_html_path = Path("static/user.html")
        if user_html_path.exists():
            return HTMLResponse(user_html_path.read_text())
        else:
            return HTMLResponse(
                """
                <!DOCTYPE html>
                <html>
                <head><title>User Interface Not Found</title></head>
                <body>
                    <h1>User Interface</h1>
                    <p>Please create 'static/user.html' in your project directory.</p>
                </body>
                </html>
                """
            )
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading user interface</h1><p>{str(e)}</p>")

@router.get("/chat.html", response_class=HTMLResponse)
async def serve_chat_html():
    """Serve a full page that loads the widget"""
    try:
        chat_html_path = Path("static/chat.html")
        if chat_html_path.exists():
            return HTMLResponse(chat_html_path.read_text())
        else:
            return HTMLResponse(
                """<h1>Chat page not found</h1>"""
            )
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading chat page</h1><p>{str(e)}</p>")

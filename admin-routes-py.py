"""
routers/admin_routes.py - Admin interface endpoints
"""
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["admin"])


@router.get("/admin", response_class=HTMLResponse)
async def get_admin_interface():
    """Redirect to admin interface"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Redirecting...</title>
        <script>
            window.location.href = '/admin.html';
        </script>
    </head>
    <body>
        <p>Redirecting to admin interface...</p>
    </body>
    </html>
    """)


@router.get("/admin.html", response_class=HTMLResponse)
async def serve_admin_html():
    """Serve the admin interface HTML file"""
    try:
        admin_html_path = Path("static/admin.html")
        if admin_html_path.exists():
            return HTMLResponse(admin_html_path.read_text())
        else:
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head><title>Admin Interface Not Found</title></head>
            <body>
                <h1>Admin Interface</h1>
                <p>Please save the admin HTML interface as 'static/admin.html' in your project directory.</p>
                <p>You can access the API documentation at <a href="/docs">/docs</a></p>
            </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading admin interface</h1><p>{str(e)}</p>")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from datetime import datetime, timezone
    
    return {
        "status": "healthy",
        "version": "7.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
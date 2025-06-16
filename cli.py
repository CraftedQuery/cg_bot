"""
cli.py - Command-line interface with Rich GUI
"""
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from rich import box
import pandas as pd

## Allow running as 'python cli.py' by fixing the package content
if not __package__:
    import sys
    package_path = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_path.parent))
    __package__= package_path.name

from .config import BASE_CONFIG_DIR, DB_PATH
from .database import get_db
from .auth import get_users_db, create_user, update_user, delete_user
from .models import UserCreate
from .ingestion import ingest

# Create CLI app
app = typer.Typer(name="rag-chatbot", help="Multi-tenant RAG Chatbot CLI")
console = Console()


@app.command("dashboard")
def dashboard():
    """Launch the interactive dashboard"""
    
    def show_main_menu():
        console.clear()
        console.print(Panel.fit(
            "[bold blue]RAG Chatbot Administration Dashboard[/bold blue]",
            title="Welcome",
            border_style="blue"
        ))
        
        console.print("\n[bold]Available Commands:[/bold]")
        console.print("1. 📊 View Analytics")
        console.print("2. 🏢 Manage Tenants & Agents")
        console.print("3. 👥 Manage Users")
        console.print("4. 📁 Ingest Content")
        console.print("5. ⚙️  System Status")
        console.print("6. 🚪 Exit")
        
        choice = typer.prompt("\nSelect an option")
        return choice
    
    def show_analytics():
        console.clear()
        console.print(Panel.fit("Analytics Dashboard", border_style="green"))
        
        # Get available tenants
        tenants = []
        for tenant_dir in BASE_CONFIG_DIR.iterdir():
            if tenant_dir.is_dir():
                tenants.append(tenant_dir.name)
        
        if not tenants:
            console.print("[yellow]No tenants found[/yellow]")
            typer.prompt("Press Enter to continue")
            return
        
        # Show tenant analytics
        for tenant in tenants:
            with get_db() as con:
                cursor = con.execute(
                    "SELECT COUNT(*) as total, COUNT(DISTINCT session_id) as sessions FROM chat_logs WHERE tenant = ?",
                    (tenant,)
                )
                total, sessions = cursor.fetchone()
            
            table = Table(title=f"Tenant: {tenant}")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("Total Queries", str(total))
            table.add_row("Unique Sessions", str(sessions))
            console.print(table)
        
        typer.prompt("\nPress Enter to continue")
    
    def show_tenants():
        console.clear()
        console.print(Panel.fit("Tenant & Agent Management", border_style="yellow"))
        
        table = Table()
        table.add_column("Tenant")
        table.add_column("Agents")
        table.add_column("Vector Store")
        
        for tenant_dir in BASE_CONFIG_DIR.iterdir():
            if tenant_dir.is_dir():
                agents = []
                for config_file in tenant_dir.iterdir():
                    if config_file.is_file() and config_file.suffix == ".json":
                        agents.append(config_file.stem)
                
                from .config import store_path
                store_exists = "✅" if store_path(tenant_dir.name, agents[0] if agents else "default").exists() else "❌"
                table.add_row(tenant_dir.name, ", ".join(agents), store_exists)
        
        console.print(table)
        typer.prompt("\nPress Enter to continue")
    
    def show_users():
        while True:
            console.clear()
            console.print(Panel.fit("User Management", border_style="red"))
            
            users_db = get_users_db()
            
            table = Table()
            table.add_column("Username")
            table.add_column("Tenant")
            table.add_column("Role")
            table.add_column("Status")
            
            for username, user_data in users_db.items():
                status = "❌ Disabled" if user_data.get("disabled", False) else "✅ Active"
                table.add_row(
                    username,
                    user_data.get("tenant", "N/A"),
                    user_data.get("role", "user"),
                    status
                )
            
            console.print(table)
            
            console.print("\n[bold]User Management Options:[/bold]")
            console.print("1. ➕ Create New User")
            console.print("2. ✏️  Edit User")
            console.print("3. ❌ Delete User")
            console.print("4. 🔄 Enable/Disable User")
            console.print("5. 🔙 Back to Main Menu")
            
            choice = typer.prompt("\nSelect an option")
            
            if choice == "1":
                # Create new user
                console.print("\n[bold green]Create New User[/bold green]")
                username = typer.prompt("Enter username")
                
                if username in users_db:
                    console.print(f"[red]❌ User '{username}' already exists[/red]")
                    typer.prompt("Press Enter to continue")
                    continue
                
                password = typer.prompt("Enter password", hide_input=True)
                confirm_password = typer.prompt("Confirm password", hide_input=True)
                
                if password != confirm_password:
                    console.print("[red]❌ Passwords don't match[/red]")
                    typer.prompt("Press Enter to continue")
                    continue
                
                tenant = typer.prompt("Enter tenant (* for all tenants)")
                role = typer.prompt("Enter role", default="user")
                
                user_data = UserCreate(
                    username=username,
                    password=password,
                    tenant=tenant,
                    role=role,
                    disabled=False
                )
                
                if create_user(user_data):
                    console.print(f"[green]✅ User '{username}' created successfully[/green]")
                else:
                    console.print(f"[red]❌ Failed to create user '{username}'[/red]")
                
                typer.prompt("Press Enter to continue")
            
            elif choice == "5":
                break
    
    def show_ingest():
        console.clear()
        console.print(Panel.fit("Content Ingestion", border_style="cyan"))
        
        console.print("[bold]Available Options:[/bold]")
        console.print("1. Ingest from Google Drive")
        console.print("2. Ingest from Sitemap")
        console.print("3. Ingest from Local Files")
        console.print("4. Back to Main Menu")
        
        choice = typer.prompt("Select an option")
        
        if choice == "1":
            tenant = typer.prompt("Enter tenant name")
            agent = typer.prompt("Enter agent name")
            folder_id = typer.prompt("Enter Google Drive folder ID")
            
            try:
                with console.status("Ingesting from Google Drive..."):
                    ingest(tenant, agent, drive=folder_id, console=console)
                console.print("[green]✅ Ingestion completed successfully[/green]")
            except Exception as e:
                console.print(f"[red]❌ Error: {str(e)}[/red]")
        
        elif choice == "2":
            tenant = typer.prompt("Enter tenant name")
            agent = typer.prompt("Enter agent name")
            sitemap_url = typer.prompt("Enter sitemap URL")
            
            try:
                with console.status("Ingesting from sitemap..."):
                    ingest(tenant, agent, sitemap=sitemap_url, console=console)
                console.print("[green]✅ Ingestion completed successfully[/green]")
            except Exception as e:
                console.print(f"[red]❌ Error: {str(e)}[/red]")
        
        elif choice == "3":
            tenant = typer.prompt("Enter tenant name")
            agent = typer.prompt("Enter agent name")
            file_path = typer.prompt("Enter file path")
            
            try:
                files = [Path(file_path)]
                with console.status("Ingesting local files..."):
                    ingest(tenant, agent, files=files, console=console)
                console.print("[green]✅ Ingestion completed successfully[/green]")
            except Exception as e:
                console.print(f"[red]❌ Error: {str(e)}[/red]")
        
        if choice != "4":
            typer.prompt("Press Enter to continue")
    
    def show_status():
        console.clear()
        console.print(Panel.fit("System Status", border_style="magenta"))
        
        # Check database
        db_status = "✅ Connected" if DB_PATH.exists() else "❌ Not Found"
        
        # Check API keys
        import os
        openai_key = "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Missing"
        google_creds = "✅ Set" if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") else "❌ Missing"
        
        # Count total interactions
        total_interactions = 0
        if DB_PATH.exists():
            with get_db() as con:
                cursor = con.execute("SELECT COUNT(*) FROM chat_logs")
                total_interactions = cursor.fetchone()[0]
        
        table = Table(title="System Status")
        table.add_column("Component")
        table.add_column("Status")
        table.add_row("Database", db_status)
        table.add_row("OpenAI API Key", openai_key)
        table.add_row("Google Credentials", google_creds)
        table.add_row("Total Interactions", str(total_interactions))
        
        console.print(table)
        typer.prompt("\nPress Enter to continue")
    
    # Main dashboard loop
    while True:
        try:
            choice = show_main_menu()
            
            if choice == "1":
                show_analytics()
            elif choice == "2":
                show_tenants()
            elif choice == "3":
                show_users()
            elif choice == "4":
                show_ingest()
            elif choice == "5":
                show_status()
            elif choice == "6":
                console.print("[green]Goodbye![/green]")
                break
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
                typer.prompt("Press Enter to continue")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            typer.prompt("Press Enter to continue")


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload")
):
    """Start the web server"""
    import uvicorn
    
    console.print(f"🚀 Starting server on {host}:{port}")
    console.print(f"📊 Dashboard: http://{host}:{port}/docs")
    console.print(f"🤖 Widget: http://{host}:{port}/widget.js")
    console.print(f"🔧 Admin: http://{host}:{port}/admin.html")
    
    module = f"{__package__}.main:app" if __package__ else "main:app"
    uvicorn.run(
        module,
        host=host,
        port=port,
        reload=reload
    )


@app.command("ingest")
def cli_ingest(
    tenant: str = typer.Argument(help="Tenant name"),
    agent: str = typer.Argument(help="Agent name"),
    sitemap: Optional[str] = typer.Option(None, "--sitemap", "-s", help="Sitemap URL"),
    drive: Optional[str] = typer.Option(None, "--drive", "-d", help="Google Drive folder ID"),
    files: Optional[List[str]] = typer.Option(None, "--file", "-f", help="Local file paths")
):
    """Ingest content into vector store"""
    
    file_paths = [Path(f) for f in files] if files else None
    
    try:
        ingest(tenant, agent, sitemap=sitemap, drive=drive, files=file_paths, console=console)
        console.print("[green]✅ Ingestion completed successfully[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("create-user")
def create_user_cli(
    username: str = typer.Argument(help="Username"),
    password: str = typer.Argument(help="Password"),
    tenant: str = typer.Option("*", help="Tenant (use '*' for all tenants)"),
    role: str = typer.Option("user", help="User role (user/admin/system_admin)"),
    agents: Optional[List[str]] = typer.Option(None, help="Agent names (comma separated)")
):
    """Create a new user"""
    
    user_data = UserCreate(
        username=username,
        password=password,
        tenant=tenant,
        role=role,
        agents=agents or []
    )
    
    if create_user(user_data):
        console.print(f"[green]✅ User '{username}' created successfully[/green]")
    else:
        console.print(f"[red]❌ User '{username}' already exists[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
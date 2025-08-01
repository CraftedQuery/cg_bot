"""
ingestion.py - Document ingestion and processing
"""
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel

from .vectorstore import update_vector_store, chunk_text
from .utils.google_drive import list_drive_files, download_drive_file
from .utils.web_scraper import parse_sitemap, download_page
from .utils.file_processors import process_file


def ingest(
    tenant: str,
    agent: str,
    sitemap: Optional[str] = None,
    drive: Optional[str] = None,
    files: Optional[List[Path]] = None,
    console: Optional[Console] = None,
    embedding_provider: str = "openai",
    embedding_model: str | None = None,
):
    """
    Ingest content from various sources into the vector store.

    Returns a dictionary mapping filenames to a boolean indicating whether OCR
    was used during processing.

    Args:
        tenant: Tenant identifier
        agent: Agent identifier
        sitemap: Optional URL to sitemap
        drive: Optional Google Drive folder ID
        files: Optional list of local files to process
        console: Optional Rich console for progress display
    """
    texts, metadatas = [], []
    ocr_info: dict[str, bool] = {}
    
    if console:
        console.print(Panel.fit(
            f"Starting ingestion for tenant [bold]{tenant}[/bold], agent [bold]{agent}[/bold]"
        ))
    
    # Process Google Drive files
    if drive:
        texts_drive, metas_drive, drive_ocr = _ingest_from_drive(drive, console)
        texts.extend(texts_drive)
        metadatas.extend(metas_drive)
        ocr_info.update(drive_ocr)
    
    # Process local files
    if files:
        texts_files, metas_files, file_ocr = _ingest_from_files(
            files,
            tenant,
            agent,
            embedding_provider,
            console,
        )
        texts.extend(texts_files)
        metadatas.extend(metas_files)
        ocr_info.update(file_ocr)
    
    # Process sitemap URLs
    if sitemap:
        texts_sitemap, metas_sitemap = _ingest_from_sitemap(sitemap, console)
        texts.extend(texts_sitemap)
        metadatas.extend(metas_sitemap)
    
    if not texts:
        msg = "Nothing to ingest"
        if console:
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            print(msg)
        return {}
    
    # Create vector store
    if console:
        console.print(f"Processing [bold]{len(texts)}[/bold] text chunks into vector store...")
    
    update_vector_store(
        tenant,
        agent,
        texts,
        metadatas,
        provider=embedding_provider,
        model=embedding_model,
    )
    
    msg = f"Vector store created/updated successfully"
    if console:
        console.print(f"[green]{msg}[/green]")
    else:
        print(msg)

    return ocr_info


def _ingest_from_drive(folder_id: str, console: Optional[Console] = None) -> tuple:
    """Ingest files from Google Drive"""
    texts, metadatas = [], []
    ocr_info: dict[str, bool] = {}
    
    try:
        files_list = list_drive_files(folder_id)
        
        if console:
            with Progress() as progress:
                task = progress.add_task(
                    f"Processing [bold]{len(files_list)}[/bold] Drive files...",
                    total=len(files_list)
                )
                for file_info in files_list:
                    progress.update(task, advance=1, description=f"Processing {file_info['name']}")
                    with download_drive_file(file_info["id"]) as file_path:
                        chunks, metas, used_ocr = process_file(file_path, file_info["name"])
                        texts.extend(chunks)
                        metadatas.extend(metas)
                        ocr_info[file_info["name"]] = used_ocr
        else:
            for file_info in files_list:
                with download_drive_file(file_info["id"]) as file_path:
                    chunks, metas, used_ocr = process_file(file_path, file_info["name"])
                    texts.extend(chunks)
                    metadatas.extend(metas)
                    ocr_info[file_info["name"]] = used_ocr
                    
    except Exception as e:
        if console:
            console.print(f"[red]Error processing Google Drive: {str(e)}[/red]")
        else:
            print(f"Error processing Google Drive: {str(e)}")
    
    return texts, metadatas, ocr_info


def _ingest_from_files(
    files: List[Path],
    tenant: str,
    agent: str,
    provider: str,
    console: Optional[Console] = None,
) -> tuple:
    """Ingest local files"""
    texts, metadatas = [], []
    ocr_info: dict[str, bool] = {}
    
    if console:
        with Progress() as progress:
            task = progress.add_task(
                f"Processing [bold]{len(files)}[/bold] local files...",
                total=len(files)
            )
            for file_path in files:
                progress.update(task, advance=1, description=f"Processing {file_path.name}")
                chunks, metas, used_ocr = process_file(file_path, file_path.name)
                texts.extend(chunks)
                metadatas.extend(metas)
                ocr_info[file_path.name] = used_ocr
    else:
        for file_path in files:
            chunks, metas, used_ocr = process_file(file_path, file_path.name)
            texts.extend(chunks)
            metadatas.extend(metas)
            ocr_info[file_path.name] = used_ocr

    return texts, metadatas, ocr_info


def _ingest_from_sitemap(sitemap_url: str, console: Optional[Console] = None) -> tuple:
    """Ingest content from sitemap URLs"""
    texts, metadatas = [], []
    
    try:
        urls = parse_sitemap(sitemap_url)
        
        if console:
            console.print(f"Found [bold]{len(urls)}[/bold] URLs in sitemap")
            with Progress() as progress:
                task = progress.add_task("Processing URLs...", total=len(urls))
                for url in urls:
                    try:
                        progress.update(task, advance=1, description=f"Processing {url}")
                        page_text = download_page(url)
                        chunks = chunk_text(page_text)
                        texts.extend(chunks)
                        metadatas.extend([{"source": url} for _ in chunks])
                    except Exception as e:
                        progress.console.print(f"Error processing {url}: {str(e)}", style="red")
        else:
            for url in urls:
                try:
                    page_text = download_page(url)
                    chunks = chunk_text(page_text)
                    texts.extend(chunks)
                    metadatas.extend([{"source": url} for _ in chunks])
                except Exception:
                    pass
                    
    except Exception as e:
        if console:
            console.print(f"[red]Error processing sitemap: {str(e)}[/red]")
        else:
            print(f"Error processing sitemap: {str(e)}")
    
    return texts, metadatas

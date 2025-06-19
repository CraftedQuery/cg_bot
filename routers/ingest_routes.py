"""
routers/ingest_routes.py - Document ingestion endpoints
"""

from pathlib import Path
from typing import List

from fastapi import APIRouter, Query, Depends, HTTPException, File, UploadFile

from ..models import User
from ..auth import get_admin_user
from ..config import DEFAULT_TENANT, DEFAULT_AGENT, upload_path
from ..ingestion import ingest
from ..vectorstore import clear_cache

router = APIRouter(tags=["ingestion"])


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    current_user: User = Depends(get_admin_user),
):
    """Upload and ingest files into the vector store"""

    # Check tenant access
    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403, detail="You don't have access to this tenant"
        )

    saved_files = []
    try:
        # Save uploaded files to tenant/agent directory
        up_dir = upload_path(tenant, agent)
        up_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            dest = up_dir / file.filename

            with dest.open("wb") as buffer:
                content = await file.read()
                buffer.write(content)

            saved_files.append(dest)

        # Ingest the files
        ingest(tenant, agent, files=saved_files)

        # Clear vector store cache to force reload
        clear_cache(tenant, agent)

        return {
            "message": f"Successfully uploaded and processed {len(files)} files",
            "files": [f.filename for f in files],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
    finally:
        pass


@router.post("/ingest/sitemap")
async def ingest_sitemap(
    sitemap_url: str,
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    current_user: User = Depends(get_admin_user),
):
    """Ingest content from a sitemap URL"""

    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403, detail="You don't have access to this tenant"
        )

    try:
        ingest(tenant, agent, sitemap=sitemap_url)

        # Clear cache
        clear_cache(tenant, agent)

        return {"message": f"Successfully ingested content from sitemap: {sitemap_url}"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error ingesting sitemap: {str(e)}"
        )


@router.post("/ingest/drive")
async def ingest_drive(
    folder_id: str,
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    current_user: User = Depends(get_admin_user),
):
    """Ingest content from a Google Drive folder"""

    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403, detail="You don't have access to this tenant"
        )

    try:
        ingest(tenant, agent, drive=folder_id)

        # Clear cache
        clear_cache(tenant, agent)

        return {
            "message": f"Successfully ingested content from Google Drive folder: {folder_id}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error ingesting Google Drive content: {str(e)}"
        )

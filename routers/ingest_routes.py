"""
routers/ingest_routes.py - Document ingestion endpoints
"""

from typing import List

from fastapi import (
    APIRouter,
    Query,
    Depends,
    HTTPException,
    File,
    UploadFile,
    Header,
)
from fastapi.responses import FileResponse

from ..models import User
from ..auth import get_admin_user, get_files_user
from ..config import DEFAULT_TENANT, DEFAULT_AGENT, uploads_path
from ..ingestion import ingest
from ..vectorstore import clear_cache
from ..database import (
    record_file_upload,
    update_file_status,
    set_file_ocr_used,
    set_file_template,
    list_uploaded_files,
    delete_uploaded_file,
    get_uploaded_file,
)

router = APIRouter(tags=["ingestion"])


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    replace: bool = Query(False, description="Replace existing files if true"),
    current_user: User = Depends(get_files_user)
):
    """Upload and ingest files into the vector store"""
    
    # Check tenant access
    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this tenant"
        )
    
    temp_files = []
    file_ids = []
    file_map = {}
    upload_dir = uploads_path(tenant, agent)
    upload_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Save uploaded files and record in DB
        for file in files:
            dest_path = upload_dir / file.filename
            if dest_path.exists() and not replace:
                raise HTTPException(
                    status_code=409,
                    detail=f"File '{file.filename}' already exists. Choose <O> for overwrite or <A> for abort."
                )

            if dest_path.exists() and replace:
                dest_path.unlink()
                from ..database import delete_uploaded_file_by_name
                delete_uploaded_file_by_name(tenant, agent, file.filename)

            with dest_path.open("wb") as buffer:
                content = await file.read()
                buffer.write(content)

            temp_files.append(dest_path)
            fid = record_file_upload(tenant, agent, file.filename, dest_path.stat().st_size)
            file_ids.append(fid)
            file_map[file.filename] = fid
        
        # Ingest the files
        ocr_flags = ingest(tenant, agent, files=temp_files)

        for fname, fid in file_map.items():
            update_file_status(fid, "ready")
            if fname in ocr_flags:
                set_file_ocr_used(fid, ocr_flags[fname])

        # Clear vector store cache to force reload
        clear_cache(tenant, agent)

        return {
            "message": f"Successfully uploaded and processed {len(files)} files",
            "files": [f.filename for f in files]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing files: {str(e)}"
        )
    finally:
        pass


@router.post("/ingest/sitemap")
async def ingest_sitemap(
    sitemap_url: str,
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    current_user: User = Depends(get_admin_user)
):
    """Ingest content from a sitemap URL"""
    
    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this tenant"
        )
    
    try:
        ingest(tenant, agent, sitemap=sitemap_url)
        
        # Clear cache
        clear_cache(tenant, agent)
        
        return {"message": f"Successfully ingested content from sitemap: {sitemap_url}"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting sitemap: {str(e)}"
        )


@router.post("/ingest/drive")
async def ingest_drive(
    folder_id: str,
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    current_user: User = Depends(get_admin_user)
):
    """Ingest content from a Google Drive folder"""
    
    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this tenant"
        )
    
    try:
        ingest(tenant, agent, drive=folder_id)
        
        # Clear cache
        clear_cache(tenant, agent)
        
        return {"message": f"Successfully ingested content from Google Drive folder: {folder_id}"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting Google Drive content: {str(e)}"
        )


@router.get("/files")
async def get_files(
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    current_user: User = Depends(get_files_user),
):
    """List uploaded files for a tenant/agent"""

    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(status_code=403, detail="You don't have access to this tenant")

    rows = list_uploaded_files(tenant, agent)
    return [
        {
            "id": r[0],
            "filename": r[1],
            "size": r[2],
            "uploaded_at": r[3],
            "status": r[4],
            "ocr_used": bool(r[5]),
            "template": bool(r[6]),
        }
        for r in rows
    ]


@router.delete("/files/{file_id}")
async def remove_file(
    file_id: int,
    current_user: User = Depends(get_files_user),
):
    """Delete an uploaded file"""

    info = get_uploaded_file(file_id)
    if not info:
        raise HTTPException(status_code=404, detail="File not found")

    tenant, agent, filename = info

    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(status_code=403, detail="You don't have access to this tenant")

    path = uploads_path(tenant, agent) / filename
    if path.exists():
        path.unlink()

    delete_uploaded_file(file_id)
    return {"message": "File deleted"}


@router.post("/files/{file_id}/template")
async def update_file_template(
    file_id: int,
    template: bool,
    current_user: User = Depends(get_files_user),
):
    """Set or unset the template flag for a file"""

    info = get_uploaded_file(file_id)
    if not info:
        raise HTTPException(status_code=404, detail="File not found")

    tenant, agent, _ = info

    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(status_code=403, detail="You don't have access to this tenant")

    set_file_template(file_id, template)
    return {"message": "Template flag updated"}


@router.get("/uploaded/{tenant}/{agent}/{filename}")
async def serve_uploaded_file(
    tenant: str,
    agent: str,
    filename: str,
    token: str | None = Query(None),
    authorization: str | None = Header(None),
):
    """Serve an uploaded file."""
    from ..auth import user_from_token

    token_value = None
    if authorization and authorization.lower().startswith("bearer "):
        token_value = authorization.split(" ", 1)[1]
    elif token:
        token_value = token

    if not token_value:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = user_from_token(token_value)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Enforce file permissions similar to get_files_user
    if user.role not in ["admin", "system_admin"] and not getattr(user, "allow_files", False):
        raise HTTPException(
            status_code=403,
            detail="Your permissions are not sufficient to complete this action",
        )

    if user.tenant != "*" and user.tenant != tenant:
        raise HTTPException(status_code=403, detail="You don't have access to this tenant")

    path = uploads_path(tenant, agent) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path)

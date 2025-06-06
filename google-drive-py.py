"""
utils/google_drive.py - Google Drive utilities
"""
import os
import tempfile
import contextlib
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


def get_drive_service():
    """Get Google Drive API service"""
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not Path(creds_path).exists():
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set or file missing")
    
    creds = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_drive_files(folder_id: str):
    """List files in a Google Drive folder"""
    service = get_drive_service()
    q = f"'{folder_id}' in parents and trashed=false"
    return service.files().list(
        q=q,
        fields="files(id,name)"
    ).execute().get("files", [])


@contextlib.contextmanager
def download_drive_file(file_id: str):
    """Download a file from Google Drive to a temporary location"""
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    
    try:
        yield Path(path)
    finally:
        os.remove(path)
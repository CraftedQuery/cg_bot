"""
rag_chatbot.py ────────────────────────────────────────────────────────────────
RAG chatbot server **v6 (patched)** – multi‑tenant, multi‑agent
───────────────────────────────────────────────────────────────────────────────
(Only change: fixed truncated block after /config route and added missing
widget.js endpoint + CLI entry. No functional logic altered.)
"""

from __future__ import annotations
import os, sys, json, tempfile, contextlib, xml.etree.ElementTree as ET
import sqlite3, datetime
from pathlib import Path
from typing import List, Optional, Dict

import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ────────────────────────────────────────────────────────────
# Paths & constants
# ────────────────────────────────────────────────────────────
BASE_CONFIG_DIR = Path("configs")
BASE_STORE_DIR = Path("vector_store")
BASE_CONFIG_DIR.mkdir(exist_ok=True)

DEFAULT_TENANT = "public"
DEFAULT_AGENT = "default"

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

# ────────────────────────────────────────────────────────────
# SQLite logger
# ────────────────────────────────────────────────────────────
DB_PATH = Path("chat_logs.db")

def _init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS chat_logs (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts TEXT,
                   tenant TEXT,
                   agent TEXT,
                   session_id TEXT,
                   question TEXT,
                   answer TEXT,
                   sources TEXT
               )"""
        )
_init_db()

# ────────────────────────────────────────────────────────────
# Helpers – config & store paths
# ────────────────────────────────────────────────────────────

def cfg_path(tenant: str, agent: str) -> Path:
    return BASE_CONFIG_DIR / tenant / f"{agent}.json"

def store_path(tenant: str, agent: str) -> Path:
    return BASE_STORE_DIR / tenant / agent

def load_config(tenant: str, agent: str) -> Dict[str, object]:
    p = cfg_path(tenant, agent)
    if p.exists():
        return json.loads(p.read_text())
    p.parent.mkdir(parents=True, exist_ok=True)
    default_cfg = {
        "bot_name": f"{tenant}-{agent}-Bot",
        "system_prompt": "You are a helpful assistant.",
        "primary_color": "#1E88E5",
        "secondary_color": "#FFFFFF",
        "avatar_url": "",
        "mode": "inline",
        "auto_open": False
    }
    p.write_text(json.dumps(default_cfg, indent=2))
    return default_cfg

# ────────────────────────────────────────────────────────────
# Google Drive helpers
# ────────────────────────────────────────────────────────────

def _drive_service():
    creds = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _drive_list_files(folder_id: str):
    drive = _drive_service()
    q = f"'{folder_id}' in parents and trashed=false"
    results = drive.files().list(q=q, fields="files(id,name,mimeType)").execute()
    return results.get("files", [])

@contextlib.contextmanager
def _drive_download(file_id: str):
    request = _drive_service().files().get_media(fileId=file_id)
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

# ────────────────────────────────────────────────────────────
# Sitemap helpers
# ────────────────────────────────────────────────────────────

def _parse_sitemap(url: str) -> List[str]:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"sm": root.tag.split("}")[0].strip("{")}
    loc_elems = root.findall("sm:url/sm:loc", ns) or root.findall("sm:sitemap/sm:loc", ns)
    urls = [e.text.strip() for e in loc_elems]
    if root.find("sm:sitemap", ns) is not None:
        nested = []
        for sub in urls:
            nested.extend(_parse_sitemap(sub))
        return nested
    return urls

def _download_page(url: str) -> str:
    resp = requests.get(url, timeout=15, headers={"User-Agent": "CQBotCrawler/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)

# ────────────────────────────────────────────────────────────
# Ingest (multi‑tenant)
# ────────────────────────────────────────────────────────────

def ingest(tenant: str, agent: str, sitemap_url: Optional[str], drive_folder: Optional[str]):
    texts, metas = [], []
    embeddings = OpenAIEmbeddings()

    if drive_folder:
        for f in _drive_list_files(drive_folder):
            with _drive_download(f["id"]) as p:
                ext = p.suffix.lower()
                if ext == ".pdf":
                    import pypdf
                    raw = "\n".join(pg.extract_text() or "" for pg in pypdf.PdfReader(str(p)).pages)
                elif ext in {".txt", ".md"}:
                    raw = p.read_text()
                else:
                    continue
                for c in TEXT_SPLITTER.split_text(raw):
                    texts.append(c); metas.append({"source": f["name"]})

    if sitemap_url:
        for u in _parse_sitemap(sitemap_url):
            try:
                txt = _download_page(u)
                for c in TEXT_SPLITTER.split_text(txt):
                    texts.append(c); metas.append({"source": u})
            except Exception:
                pass

    if not texts:
        print("Nothing to ingest"); return

    path = store_path(tenant, agent)
    path.mkdir(parents=True, exist_ok=True)
    FAISS.from_texts(texts, embeddings, metadatas=metas).save_local(str(path))
    print("Vector store written →", path)

# ────────────────────────────────────────────────────────────
# FastAPI backend
# ────────────────────────────────────────────────────────────

app = FastAPI(title="CraftedQuery Multi‑Tenant RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    messages: List[dict]

_vec_cache: Dict[str, FAISS] = {}

def _load_db(tenant: str, agent: str):
    key = f"{tenant}/{agent}"
    if key in _vec_cache:
        return _vec_cache[key]
    p = store_path(tenant, agent)
    if not p.exists():
        raise HTTPException(404, f"vector store for {key} not found; run ingest")
    db = FAISS.load_local(str(p), OpenAIEmbeddings())
    _vec_cache[key] = db
    return db

@app.post("/chat")
async def chat(req: ChatRequest, request: Request,
               tenant: str = Query(DEFAULT_TENANT),
               agent: str = Query(DEFAULT_AGENT)):
    db = _load_db(tenant, agent)
    cfg = load_config(tenant, agent)

    user_msg = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
    docs_scores = db.similarity_search_with_score(user_msg, k=4)
    context = "\n".join([d.page_content for d, _ in docs_scores])

    system_msg = cfg["system_prompt"] + "\nContext:\n" + context
    full_msgs = [{"role": "system", "content": system_msg}] + req.messages

    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    resp = openai.ChatCompletion.create(model="gpt-4o-mini", temperature=0.3, messages=full_msgs)
    answer = resp.choices[0].message["content"]

    sources, seen = [], set()
    for d, _ in docs_scores:
        src = d.metadata.get("source", "")
        if src and src not in seen:
            sources.append({"source": src}); seen.add(src)

    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO chat_logs(ts,tenant,agent,session_id,question,answer,sources) VALUES (?,?,?,?,?,?,?)",
                    (datetime.datetime.utcnow().isoformat(), tenant, agent,
                     request.headers.get("X-Session-Id", "anon"), user_msg, answer, json.dumps(sources)))

    return {"reply": answer, "sources": sources}

# ────────────────────────────────────────────────────────────
# Config & widget endpoints
# ────────────────────────────────────────────────────────────

WIDGET_JS = r"""/* lightweight widget, same as earlier versions */
(function(){function u(){return([...Array(30)]).map(()=>Math.random().toString(36)[2]).join('');}const sid=sessionStorage.getItem('cq_sid')||(()=>{const id=u();sessionStorage.setItem('cq_sid',id);return id;})();function $(i){return document.getElementById(i);}function el(t,c){const e=document.createElement(t);if(c)e.className=c;return e;}let tenant="";let agent="";function qs(){const p=new URLSearchParams(location.search);tenant=p.get('tenant')||'public';agent=p.get('agent')||'default';}qs();const msgs=[];async function send(q){msgs.push({role:'user',content:q});const r=await fetch(`/chat?tenant=${tenant}&agent=${agent}`,{method:'POST',headers:{'Content-Type':'application/json','X-Session-Id':sid},body:JSON.stringify({messages:msgs})});const d=await r.json();msgs.push({role:'assistant',content:d.reply});addMsg('assistant',d.reply,d.sources);}function addMsg(role,txt,srcs){const p=$('cq_chat_pane');const w=el('div',role);w.appendChild(document.createTextNode(txt));if(srcs){const det=el('details','src');det.innerHTML='<summary>Sources</summary>';srcs.forEach(s=>{const d=el('div');d.textContent=s.source;det.appendChild(d);});w.appendChild(det);}p.appendChild(w);p.scrollTop=p.scrollHeight;}async function init(){const cfg=await fetch(`/config?tenant=${tenant}&agent=${agent}`).then(r=>r.json());const root;if(cfg.mode==='bubble'){root=el('div');root.id='cq_chat_container';root.style.cssText='position:fixed;bottom:90px;right:24px;max-width:320px;display:none;background:white;border:1px solid #ccc;border-radius:8px;z-index:9999';document.body.appendChild(root);const btn=el('button');btn.textContent='💬';btn.style.cssText=`position:fixed;bottom:24px;right:24px;width:56px;height:56px;border-radius:50%;background:${cfg.primary_color};color:${cfg.secondary_color};border:none;cursor:pointer;z-index:9999`;btn.onclick=()=>{root.style.display=root.style.display==='none'?'block':'none';};document.body.appendChild(btn);} else {root=$('cq_chat');}root.innerHTML=`<style>#cq_chat_pane{height:300px;overflow:auto;background:${cfg.primary_color}20;border-radius:8px;padding:8px;font-family:sans-serif}#cq_chat_input{width:100%;padding:6px;border:1px solid ${cfg.primary

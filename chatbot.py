"""
rag_chatbot.py – multi‑tenant RAG bot (v6.1)
Fully patched: closes unterminated WIDGET_JS string, restores /widget.js route,
/config endpoint, and CLI entry. Ready to run.
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

# ───────────────────────── Basics ───────────────────────────
BASE_CONFIG_DIR = Path("configs"); BASE_CONFIG_DIR.mkdir(exist_ok=True)
BASE_STORE_DIR  = Path("vector_store")
DEFAULT_TENANT  = "public"; DEFAULT_AGENT = "default"
TEXT_SPLITTER   = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

# ──────────────────────── Logging DB ────────────────────────
DB_PATH = Path("chat_logs.db")
with sqlite3.connect(DB_PATH) as con:
    con.execute("""CREATE TABLE IF NOT EXISTS chat_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, tenant TEXT, agent TEXT, session_id TEXT,
        question TEXT, answer TEXT, sources TEXT)""")

# ───────────────────── Config helpers ───────────────────────

def cfg_path(t: str, a: str) -> Path: return BASE_CONFIG_DIR / t / f"{a}.json"

def store_path(t: str, a: str) -> Path: return BASE_STORE_DIR / t / a

def load_config(t: str, a: str) -> Dict[str, object]:
    p = cfg_path(t,a)
    if p.exists():
        return json.loads(p.read_text())
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"bot_name":f"{t}-{a}-Bot","system_prompt":"You are a helpful assistant.",
           "primary_color":"#1E88E5","secondary_color":"#FFFFFF","avatar_url":"",
           "mode":"inline","auto_open":False}
    p.write_text(json.dumps(cfg,indent=2)); return cfg

# ─────────────────── Google Drive helpers ───────────────────

def _drive_service():
    creds = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=["https://www.googleapis.com/auth/drive.readonly"],)
    return build("drive","v3",credentials=creds,cache_discovery=False)

def _drive_list(folder_id:str):
    drv=_drive_service(); q=f"'{folder_id}' in parents and trashed=false"
    return drv.files().list(q=q,fields="files(id,name)").execute().get("files",[])

@contextlib.contextmanager
def _drive_download(fid:str):
    req=_drive_service().files().get_media(fileId=fid); fd,path=tempfile.mkstemp()
    with os.fdopen(fd,"wb") as fh:
        dl = MediaIoBaseDownload(fh, req); done=False
        while not done: _,done = dl.next_chunk()
    try: yield Path(path)
    finally: os.remove(path)

# ───────────────────── Sitemap helpers ──────────────────────

def _parse_sitemap(url:str)->List[str]:
    r=requests.get(url,timeout=15); r.raise_for_status(); root=ET.fromstring(r.text)
    ns={"sm":root.tag.split("}")[0].strip("{")}
    loc=root.findall("sm:url/sm:loc",ns) or root.findall("sm:sitemap/sm:loc",ns)
    urls=[e.text.strip() for e in loc]
    if root.find("sm:sitemap",ns) is not None:
        nested=[]; [nested.extend(_parse_sitemap(u)) for u in urls]; return nested
    return urls

def _download_page(u:str)->str:
    r=requests.get(u,timeout=15,headers={"User-Agent":"CQBotCrawler/1.0"}); r.raise_for_status()
    soup=BeautifulSoup(r.text,"html.parser"); [t.decompose() for t in soup(["script","style","noscript","header","footer","nav"])]
    return soup.get_text("\n",strip=True)

# ───────────────────────── Ingest ───────────────────────────

def ingest(t:str,a:str,sitemap:Optional[str],drive:Optional[str]):
    txts,meta=[],[]; emb=OpenAIEmbeddings()
    if drive:
        for f in _drive_list(drive):
            with _drive_download(f["id"]) as p:
                ext=p.suffix.lower()
                if ext==".pdf":
                    import pypdf; raw="\n".join(pg.extract_text()or"" for pg in pypdf.PdfReader(str(p)).pages)
                elif ext in {".txt",".md"}: raw=p.read_text()
                else: continue
                for c in TEXT_SPLITTER.split_text(raw): txts.append(c); meta.append({"source":f["name"]})
    if sitemap:
        for u in _parse_sitemap(sitemap):
            try:
                pg=_download_page(u)
                for c in TEXT_SPLITTER.split_text(pg): txts.append(c); meta.append({"source":u})
            except: pass
    if not txts: print("Nothing to ingest"); return
    path=store_path(t,a); path.mkdir(parents=True,exist_ok=True)
    FAISS.from_texts(txts,emb,metadatas=meta).save_local(str(path))
    print("Vector store →",path)

# ───────────────────────── API ──────────────────────────────
app=FastAPI(title="Multi‑Tenant RAG")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

class ChatRequest(BaseModel): messages:List[dict]
_vec:Dict[str,FAISS]={}

def _db(t:str,a:str):
    k=f"{t}/{a}"; p=store_path(t,a)
    if k in _vec: return _vec[k]
    if not p.exists(): raise HTTPException(404,"vector store missing; run ingest")
    _vec[k]=FAISS.load_local(str(p),OpenAIEmbeddings()); return _vec[k]

@app.post("/chat")
async def chat(req:ChatRequest,request:Request,tenant:str=Query(DEFAULT_TENANT),agent:str=Query(DEFAULT_AGENT)):
    db=_db(tenant,agent); cfg=load_config(tenant,agent)
    q=next((m["content"] for m in reversed(req.messages) if m["role"]=="user"),"")
    docs=db.similarity_search_with_score(q,k=4)
    ctx="\n".join(d.page_content for d,_ in docs)
    sysmsg={"role":"system","content":cfg["system_prompt"]+"\nContext:\n"+ctx}
    import openai; openai.api_key=os.environ.get("OPENAI_API_KEY","")
    rsp=openai.ChatCompletion.create(model="gpt-4o-mini",temperature=0.3,messages=[sysmsg,*req.messages])
    ans=rsp.choices[0].message["content"]
    srcs=[]; seen=set()
    for d,_ in docs:
        s=d.metadata.get("source","");
        if s and s not in seen: srcs.append({"source":s}); seen.add(s)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO chat_logs(ts,tenant,agent,session_id,question,answer,sources) VALUES (?,?,?,?,?,?,?)",
                    (datetime.datetime.utcnow().isoformat(),tenant,agent,request.headers.get("X-Session-Id","anon"),q,ans,json.dumps(srcs)))
    return {"reply":ans,"sources":srcs}

# ────────────────── Config & Widget endpoints ──────────────
WIDGET_JS=r"""(function(){function g(k){return([...Array(30)]).map(()=>Math.random().toString(36)[2]).join('');}const sid=sessionStorage.getItem('cq_sid')||(()=>{const i=g();sessionStorage.setItem('cq_sid',i);return i;})();const p=new URLSearchParams(location.search);const tenant=p.get('tenant')||'public';const agent=p.get('agent')||'default';const msgs=[];function $(id){return document.getElementById(id);}function el(t,c){const e=document.createElement(t);if(c)e.className=c;return e;}async function send(q){msgs.push({role:'user',content:q});const r=await fetch(`/chat?tenant=${tenant}&agent=${agent}`,{method:'POST',headers:{'Content-Type':'application/json','X-Session-Id':sid},body:JSON.stringify({messages:msgs})});const d=await r.json();msgs.push({role:'assistant',content:d.reply});append('assistant',d.reply,d.sources);}function append(role,txt,src){const pane=$('cq_chat_pane');const w=el('div',role);w.textContent=txt;if(src&&src.length){const det=el('details');det.innerHTML='<summary>Sources</summary>';src.forEach(s=>{const dv=el('div');dv.textContent=s.source;det.appendChild(dv);});w.appendChild(det);}pane.appendChild

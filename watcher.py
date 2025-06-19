"""Background ingestion watcher"""
import time
from argparse import ArgumentParser
from pathlib import Path

from .ingestion import ingest
from .config import BASE_DIR


def watch_folder(directory: Path, tenant: str, agent: str, interval: int, provider: str, model: str | None):
    processed: dict[Path, float] = {}
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        files = [f for f in directory.iterdir() if f.is_file()]
        changed = False
        for f in files:
            mtime = f.stat().st_mtime
            if processed.get(f) != mtime:
                changed = True
            processed[f] = mtime
        if changed and files:
            ingest(
                tenant,
                agent,
                files=files,
                console=None,
                embedding_provider=provider,
                embedding_model=model,
            )
        time.sleep(interval)


def main():
    ap = ArgumentParser(description="Watch a folder and ingest files when updated")
    ap.add_argument("--tenant", default="public")
    ap.add_argument("--agent", default="default")
    ap.add_argument("--uploads", type=Path, default=BASE_DIR / "uploads")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic", "vertexai", "local"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    watch_folder(args.uploads, args.tenant, args.agent, args.interval, args.provider, args.model)


if __name__ == "__main__":
    main()

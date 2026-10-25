"""Serve training runs to the dashboard.

    python serve.py            # http://localhost:8000 (API + built dashboard)

API (all JSON):
    GET /api/runs                       every run, newest first, with a summary
    GET /api/runs/<name>                config, live status, all metrics and evals
    GET /api/runs/<name>/samples/<step> every rollout from that step
    GET /api/runs/<name>/live           rollouts in flight right now, partial text included
"""

import argparse
import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"
DIST = HERE / "dashboard" / "dist"


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line still being written
    return rows


def summary(run):
    metrics = read_jsonl(run / "metrics.jsonl")
    evals = read_jsonl(run / "evals.jsonl")
    config = read_json(run / "config.json", {})
    return {
        "name": run.name,
        "started": config.get("started"),
        "steps": config.get("steps"),
        "step": metrics[-1]["step"] if metrics else 0,
        "reward": metrics[-1]["reward_mean"] if metrics else None,
        "evalFirst": evals[0]["pass1"] if evals else None,
        "evalLast": evals[-1]["pass1"] if evals else None,
        "status": read_json(run / "status.json"),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DIST), **kw)

    def end_headers(self):
        # Hashed assets can be cached forever; index.html must not be, or a
        # rebuilt dashboard keeps asking for chunks that no longer exist.
        if not self.path.startswith("/assets/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if not path.startswith("/api/"):
            # SPA fallback: unknown paths get index.html so client routing works.
            if not path.startswith("/assets/") and not (DIST / path.lstrip("/")).is_file():
                self.path = "/index.html"
            return super().do_GET()

        if path == "/api/runs":
            runs = [summary(r) for r in RUNS.iterdir() if (r / "config.json").exists()] if RUNS.exists() else []
            return self.send_json(sorted(runs, key=lambda r: r["started"] or 0, reverse=True))

        m = re.fullmatch(r"/api/runs/([\w.-]+)(?:/samples/(\d+)|/(live))?", path)
        run = RUNS / m.group(1) if m else None
        if not run or not (run / "config.json").exists():
            return self.send_json({"message": "Run not found"}, 404)
        if m.group(3):
            return self.send_json(read_json(run / "live.json"))
        if m.group(2):
            samples = read_json(run / "samples" / f"step_{int(m.group(2)):05d}.json")
            if samples is None:
                return self.send_json({"message": "No samples for that step"}, 404)
            return self.send_json(samples)
        return self.send_json({
            "name": run.name,
            "config": read_json(run / "config.json"),
            "status": read_json(run / "status.json"),
            "metrics": read_jsonl(run / "metrics.jsonl"),
            "evals": read_jsonl(run / "evals.jsonl"),
            "checks": read_jsonl(run / "checks.jsonl"),
            "selected": read_json(run / "selected.json"),
        })

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"dashboard: http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

"""Download Spider dev questions and extract the Spider test-suite databases for them.

    python scripts/prepare_spider.py

Writes data/spider/dev.json and data/spider/database/<db_id>/*.sqlite. Each dev
database directory holds the original Spider database (<db_id>.sqlite) and the
test-suite variants (Zhong et al. 2020, github.com/taoyds/test-suite-sql-eval).
"""
import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "spider"
# Google Drive file linked from the test-suite-sql-eval README (testsuitedatabases.zip, 1.27 GB).
ZIP_URL = "https://drive.usercontent.google.com/download?id=1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w&export=download&confirm=t"
ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset=xlangai/spider&config=spider&split=validation&offset={}&length=100"


def fetch_dev():
    rows, total = [], None
    while total is None or len(rows) < total:
        with urllib.request.urlopen(ROWS_URL.format(len(rows)), timeout=60) as r:
            page = json.load(r)
        total = page["num_rows_total"]
        if not page["rows"]:
            raise RuntimeError(f"Spider dev download stopped at {len(rows)} of {total} rows")
        rows += [{k: p["row"][k] for k in ("db_id", "question", "query")} for p in page["rows"]]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DATA / "download" / "testsuitedatabases.zip")
    args = ap.parse_args()
    dev_path = DATA / "dev.json"
    if not dev_path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        dev = fetch_dev()
        dev_path.write_text(json.dumps(dev, indent=1) + "\n")
    dev = json.loads(dev_path.read_text())
    db_ids = sorted({r["db_id"] for r in dev})
    if not args.zip.exists():
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.zip.with_name(args.zip.name + ".part")
        with urllib.request.urlopen(ZIP_URL, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.replace(args.zip)
    out = DATA / "database"
    with zipfile.ZipFile(args.zip) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts
            # database/<db_id>/<file>.sqlite; skip macOS resource forks.
            if len(parts) != 3 or parts[0] != "database" or parts[1] not in db_ids or not parts[2].endswith(".sqlite"):
                continue
            target = out / parts[1] / parts[2]
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    for db_id in db_ids:
        if not (out / db_id / f"{db_id}.sqlite").exists():
            raise RuntimeError(f"Missing original database for {db_id}")
    n = sum(len(list((out / d).glob("*.sqlite"))) for d in db_ids)
    print(f"{len(dev)} dev questions over {len(db_ids)} databases; {n} database files in {out}")


if __name__ == "__main__":
    main()

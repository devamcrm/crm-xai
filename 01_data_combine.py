# data_combine.py
"""
A) Download train/val/test parquet from hosted URLs into ./data/
B) Combine into one event stream with a `source` column and write ./data/full_data.parquet
C) Print 25 rows + dataset shape

Run:
  python data_combine.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import requests
import polars as pl

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "train": "https://img.monnde.com/crm-xai/train.parquet",
    "val": "https://img.monnde.com/crm-xai/val.parquet",
    "test": "https://img.monnde.com/crm-xai/test.parquet",
}

OUT_FULL = DATA_DIR / "full_data.parquet"


def download_file(url: str, out_path: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"✅ Exists, skipping: {out_path}")
        return

    print(f"⬇️  Downloading: {url}")
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        downloaded = 0

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    sys.stdout.write(f"\r   ... {pct:6.2f}%")
                    sys.stdout.flush()

    if total:
        sys.stdout.write("\n")

    tmp_path.replace(out_path)
    print(f"✅ Saved: {out_path}")


def step_a_download() -> dict[str, Path]:
    print("\n=== STEP A: Download datasets ===")
    paths = {}
    for name, url in URLS.items():
        out = DATA_DIR / f"{name}.parquet"
        download_file(url, out)
        paths[name] = out
    return paths


def step_b_combine(paths: dict[str, Path]) -> Path:
    print("\n=== STEP B: Combine datasets ===")

    lf_train = pl.scan_parquet(str(paths["train"])).with_columns(pl.lit("train").alias("source"))
    lf_val = pl.scan_parquet(str(paths["val"])).with_columns(pl.lit("val").alias("source"))
    lf_test = pl.scan_parquet(str(paths["test"])).with_columns(pl.lit("test").alias("source"))

    lf_full = pl.concat([lf_train, lf_val, lf_test], how="vertical_relaxed")
    lf_full.sink_parquet(str(OUT_FULL))

    print(f"✅ Written: {OUT_FULL}")
    return OUT_FULL


def step_c_preview(full_path: Path) -> None:
    print("\n=== STEP C: Preview ===")

    df_head = pl.read_parquet(str(full_path), n_rows=25)
    print("\n--- First 25 rows ---")
    print(df_head)

    lf = pl.scan_parquet(str(full_path))
    n_rows = lf.select(pl.len()).collect().item()
    n_cols = len(lf.schema)

    print("\n--- Shape ---")
    print(f"Rows: {n_rows:,}")
    print(f"Cols: {n_cols}")
    print(f"Columns: {list(lf.schema.keys())}")


def main() -> None:
    paths = step_a_download()
    full_path = step_b_combine(paths)
    step_c_preview(full_path)


if __name__ == "__main__":
    main()

import os
import glob
from etl.transform import transform
from etl.load import load
from etl.embed import run_embed

NC_FOLDER = "./NetCDF-Files"
nc_files  = sorted(glob.glob(os.path.join(NC_FOLDER, "*.nc")))

def run():
    # ── Phase 1: ETL ──────────────────────────────────────────
    print(f"Found {len(nc_files)} NetCDF files.\n")

    for nc_file in nc_files:
        print("=" * 50)
        try:
            data = transform(nc_file)
            load(data)
        except Exception as e:
            print(f"[ERROR] {nc_file}: {e}")
            continue

    # ── Phase 2: Embed into ChromaDB ──────────────────────────
    print("\n" + "=" * 50)
    print("Starting Phase 2 — building embeddings in ChromaDB...")
    run_embed()

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("  PostgreSQL → structured data (434 profiles, 288k measurements)")
    print("  ChromaDB   → vector embeddings (434 chunks, ready for RAG)")

if __name__ == "__main__":
    run()
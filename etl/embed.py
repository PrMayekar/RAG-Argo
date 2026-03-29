import chromadb
from chromadb.utils import embedding_functions
from db.connection import get_session
from db.models import Profile, ArgoFloat, Measurement

# ── ChromaDB client — saves to disk in ./chroma_db folder ──────────────────
CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "argo_profiles"

# Uses sentence-transformers locally — no API key needed
# Downloads all-MiniLM-L6-v2 (~90MB) on first run automatically
EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

QC_LABELS = {
    "1": "good",        "2": "probably good",
    "3": "probably bad","4": "bad",
    "8": "estimated",   "9": "missing"
}

def qc_label(flag) -> str:
    return QC_LABELS.get(str(flag).strip(), "unknown")


def build_chunk_text(wmo_id: str, platform: str,
                     profile: Profile, measurements: list) -> str:

    date_str  = profile.profile_date.strftime("%Y-%m-%d") \
                if profile.profile_date else "unknown date"
    direction = "ascending" if profile.direction == "A" else "descending"

    header = (
        f"Float {wmo_id} ({platform}), cycle {profile.cycle_number}, "
        f"{direction} profile on {date_str} at "
        f"lat {round(profile.latitude, 3)}, lon {round(profile.longitude, 3)}."
    )

    # Only use good/probably good QC measurements
    good = [m for m in measurements
            if m.temp_qc in ("1", "2") or m.sal_qc in ("1", "2")]

    if not good:
        return header + " No good quality measurements available."

    surface = good[0]
    deep    = good[-1]

    surface_line = (
        f"Surface ({surface.pressure} dbar): "
        f"temp {surface.temperature}°C ({qc_label(surface.temp_qc)}), "
        f"salinity {surface.salinity} PSU ({qc_label(surface.sal_qc)})."
    )

    deep_line = ""
    if deep.depth_level != surface.depth_level:
        deep_line = (
            f" Deep ({deep.pressure} dbar): "
            f"temp {deep.temperature}°C ({qc_label(deep.temp_qc)}), "
            f"salinity {deep.salinity} PSU ({qc_label(deep.sal_qc)})."
        )

    # Fix Bug 2: guard against empty lists before calling min/max
    temps = [m.temperature for m in good if m.temperature is not None]
    sals  = [m.salinity    for m in good if m.salinity    is not None]
    pres  = [m.pressure    for m in good if m.pressure    is not None]

    if temps and pres:
        summary = (
            f" Column: {len(good)} valid levels, "
            f"temp range {round(min(temps),2)}–{round(max(temps),2)}°C, "
        )
        if sals:
            summary += f"salinity {round(min(sals),2)}–{round(max(sals),2)} PSU, "
        else:
            summary += "salinity data unavailable, "
        summary += f"pressure {round(min(pres),1)}–{round(max(pres),1)} dbar."
    else:
        summary = f" Column: {len(good)} valid levels."

    return header + " " + surface_line + deep_line + summary


def run_embed():
    print("\n[EMBED] Connecting to ChromaDB...")

    # Persistent client — data saved to ./chroma_db on disk
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name               = COLLECTION_NAME,
        embedding_function = EMBEDDING_FN,
        metadata           = {"hnsw:space": "cosine"}
    )

    session = get_session()
    try:
        # Fetch all profiles joined with their float info
        rows = (
            session.query(Profile, ArgoFloat)
            .join(ArgoFloat, ArgoFloat.id == Profile.float_id)
            .all()
        )
        print(f"[EMBED] Found {len(rows)} profiles to process.")

        documents = []   # text chunks
        metadatas = []   # structured metadata for filtering
        ids       = []   # unique ID per chunk

        skipped = 0

        for profile, argo_float in rows:
            chunk_id = f"profile_{profile.id}"

            # Skip if already in ChromaDB
            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                skipped += 1
                continue

            # Get measurements ordered shallow → deep
            measurements = (
                session.query(Measurement)
                .filter_by(profile_id=profile.id)
                .order_by(Measurement.depth_level)
                .all()
            )

            # Build the text chunk
            text = build_chunk_text(
                argo_float.wmo_id,
                argo_float.platform_type or "unknown",
                profile,
                measurements
            )

            # Metadata — used for filtering queries later
            # e.g. "only search float 1901771" or "only 2026-01-03"
            metadata = {
                "profile_id":   profile.id,
                "float_id":     argo_float.id,
                "wmo_id":       argo_float.wmo_id,
                "platform":     argo_float.platform_type or "unknown",
                "cycle_number": profile.cycle_number,
                "date":         profile.profile_date.strftime("%Y-%m-%d")
                                if profile.profile_date else "unknown",
                "latitude":     round(profile.latitude, 4)
                                if profile.latitude else 0.0,
                "longitude":    round(profile.longitude, 4)
                                if profile.longitude else 0.0,
            }

            documents.append(text)
            metadatas.append(metadata)
            ids.append(chunk_id)

        if skipped:
            print(f"[EMBED] Skipped {skipped} already-embedded profiles.")

        if not documents:
            print("[EMBED] Nothing new to embed. All profiles already indexed.")
            return

        # Add to ChromaDB in batches of 50
        # ChromaDB handles embedding generation automatically
        print(f"[EMBED] Embedding {len(documents)} profiles "
              f"(downloading model on first run ~90MB)...")

        BATCH = 50
        for i in range(0, len(documents), BATCH):
            batch_docs  = documents[i : i + BATCH]
            batch_meta  = metadatas[i : i + BATCH]
            batch_ids   = ids[i : i + BATCH]

            collection.add(
                documents = batch_docs,
                metadatas = batch_meta,
                ids       = batch_ids,
            )
            print(f"[EMBED] Batch {i//BATCH + 1} done "
                  f"({min(i+BATCH, len(documents))}/{len(documents)})")

        print(f"\n[EMBED] Done! {len(documents)} profiles indexed in ChromaDB.")
        print(f"[EMBED] Total in collection: {collection.count()}")

        # Show a sample chunk so you can see what it looks like
        print("\n--- Sample chunk (first profile) ---")
        print(documents[0])
        print("\n--- Sample metadata ---")
        print(metadatas[0])

    except Exception as e:
        print(f"[EMBED] Error: {e}")
        raise
    finally:
        session.close()
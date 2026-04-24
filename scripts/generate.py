"""
Workshop asset generator — one script that builds everything.

Runs the full setup pipeline end-to-end:
  1. catalog  — regenerate backend/data/products.json (1000 synthetic products)
  2. embed    — load the catalog into MongoDB Atlas and generate Voyage AI
                embeddings for each product
  3. indexes  — create the Atlas Vector Search + Atlas Search indexes

Usage:
  python scripts/generate.py                      # run all stages
  python scripts/generate.py --only catalog       # just regenerate the JSON
  python scripts/generate.py --only embed         # just (re)load + embed
  python scripts/generate.py --only indexes       # just create indexes
  python scripts/generate.py --skip embed         # run catalog + indexes only
  python scripts/generate.py --force              # destructive re-run (wipes coll)

Environment:
  MONGODB_URI       required for embed + indexes
  VOYAGE_API_KEY    required for embed

This orchestrator intentionally duplicates the logic of
scripts/01_load_and_embed.py and scripts/02_create_indexes.py rather than
importing from them, so those numbered scripts stay simple pedagogical
step-through examples and are not affected by orchestrator changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "backend" / "data" / "products.json"

DB_NAME = "workshop"
COLL_NAME = "products"
EMBED_MODEL = "voyage-4-large"
EMBED_BATCH = 128
VECTOR_INDEX_NAME = "vector_index"
TEXT_INDEX_NAME = "text_search_index"

STAGES = ["catalog", "embed", "indexes"]


# ---------------------------------------------------------------------------
# Stage 1: catalog
# ---------------------------------------------------------------------------

def stage_catalog(output_path: Path = CATALOG_PATH, force: bool = False) -> dict:
    """(Re)generate the 1000-product JSON catalog."""
    if output_path.exists() and not force:
        size = output_path.stat().st_size
        print(f"[catalog] {output_path} already exists ({size:,} bytes). Use --force to overwrite.")
        with output_path.open() as f:
            existing = json.load(f)
        return {"stage": "catalog", "skipped": True, "count": len(existing), "path": str(output_path)}

    # Import lazily so the orchestrator can run other stages without this dependency.
    sys.path.insert(0, str(REPO_ROOT / "backend" / "utils"))
    try:
        from generate_catalog import generate_catalog  # type: ignore
    finally:
        sys.path.pop(0)

    print(f"[catalog] Generating products -> {output_path}")
    products = generate_catalog()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(products, f, indent=2)
    print(f"[catalog] Wrote {len(products)} products ({output_path.stat().st_size:,} bytes)")
    return {"stage": "catalog", "skipped": False, "count": len(products), "path": str(output_path)}


# ---------------------------------------------------------------------------
# Stage 2: embed
# ---------------------------------------------------------------------------

def stage_embed(catalog_path: Path = CATALOG_PATH, force: bool = False) -> dict:
    """Load the catalog into MongoDB and generate Voyage embeddings."""
    import voyageai
    from pymongo import MongoClient

    mongo_uri = os.environ.get("MONGODB_URI")
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI is not set. Add it to .env or export it.")
    if not voyage_key:
        raise RuntimeError("VOYAGE_API_KEY is not set. Add it to .env or export it.")

    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found at {catalog_path}. Run --only catalog first.")

    client = MongoClient(mongo_uri)
    try:
        coll = client[DB_NAME][COLL_NAME]
        existing = coll.count_documents({})
        if existing > 0 and not force:
            print(
                f"[embed] Collection {DB_NAME}.{COLL_NAME} already has {existing} documents. "
                "Use --force to wipe and re-embed (burns Voyage quota)."
            )
            return {"stage": "embed", "skipped": True, "count": existing}

        with catalog_path.open() as f:
            products = json.load(f)
        print(f"[embed] Loaded {len(products)} products from {catalog_path.name}")

        vo = voyageai.Client(api_key=voyage_key)
        descriptions = [p["description"] for p in products]
        embeddings: list[list[float]] = []
        for i in range(0, len(descriptions), EMBED_BATCH):
            batch = descriptions[i : i + EMBED_BATCH]
            result = vo.embed(batch, model=EMBED_MODEL, input_type="document")
            embeddings.extend(result.embeddings)
            print(f"[embed]   batch {i // EMBED_BATCH + 1} ({len(embeddings)}/{len(descriptions)})")

        for product, embedding in zip(products, embeddings):
            product["description_embedding"] = embedding

        coll.delete_many({})
        result = coll.insert_many(products)
        print(f"[embed] Inserted {len(result.inserted_ids)} documents into {DB_NAME}.{COLL_NAME}")
        return {
            "stage": "embed",
            "skipped": False,
            "count": len(result.inserted_ids),
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Stage 3: indexes
# ---------------------------------------------------------------------------

def stage_indexes(wait: bool = False, timeout_s: int = 300) -> dict:
    """Create the vector and text search indexes (idempotent)."""
    from pymongo import MongoClient
    from pymongo.errors import OperationFailure
    from pymongo.operations import SearchIndexModel

    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI is not set. Add it to .env or export it.")

    vector_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "description_embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "category"},
                {"type": "filter", "path": "price"},
            ]
        },
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
    )
    text_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "name": {"type": "string", "analyzer": "lucene.standard"},
                    "description": {"type": "string", "analyzer": "lucene.standard"},
                    "category": {"type": "stringFacet"},
                    "brand": {"type": "stringFacet"},
                },
            }
        },
        name=TEXT_INDEX_NAME,
        type="search",
    )

    client = MongoClient(mongo_uri)
    try:
        coll = client[DB_NAME][COLL_NAME]
        existing_names = {idx.get("name") for idx in coll.list_search_indexes()}
        created: list[str] = []
        skipped: list[str] = []

        for model, name in ((vector_model, VECTOR_INDEX_NAME), (text_model, TEXT_INDEX_NAME)):
            if name in existing_names:
                print(f"[indexes] {name!r} already exists, skipping")
                skipped.append(name)
                continue
            try:
                coll.create_search_index(model)
                print(f"[indexes] submitted {name!r}")
                created.append(name)
            except OperationFailure as exc:
                # Defensive: race where the index appeared between list and create.
                if "already exists" in str(exc).lower():
                    print(f"[indexes] {name!r} already exists (race), skipping")
                    skipped.append(name)
                else:
                    raise

        statuses = {idx.get("name"): idx.get("status") for idx in coll.list_search_indexes()}
        print(f"[indexes] statuses: {statuses}")

        if wait and created:
            print(f"[indexes] waiting up to {timeout_s}s for {created} to reach READY...")
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                statuses = {idx.get("name"): idx.get("status") for idx in coll.list_search_indexes()}
                if all(statuses.get(n) == "READY" for n in created):
                    print(f"[indexes] all ready: {statuses}")
                    break
                print(f"[indexes]   still building: {statuses}")
                time.sleep(10)
            else:
                print(f"[indexes] timed out waiting; final statuses: {statuses}")

        return {
            "stage": "indexes",
            "created": created,
            "skipped_existing": skipped,
            "statuses": statuses,
        }
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all(
    stages: Optional[list[str]] = None,
    force: bool = False,
    wait_indexes: bool = False,
) -> dict:
    """Run the requested stages in order and return a combined summary."""
    selected = stages or STAGES
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Known: {STAGES}")

    summary: dict = {"stages_run": selected, "force": force, "results": {}}

    for stage in selected:
        started = time.time()
        if stage == "catalog":
            summary["results"]["catalog"] = stage_catalog(force=force)
        elif stage == "embed":
            summary["results"]["embed"] = stage_embed(force=force)
        elif stage == "indexes":
            summary["results"]["indexes"] = stage_indexes(wait=wait_indexes)
        summary["results"][stage]["duration_s"] = round(time.time() - started, 2)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--only",
        type=str,
        help=f"Comma-separated list of stages to run. Default: all. Available: {','.join(STAGES)}",
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Comma-separated list of stages to skip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run destructive stages (re-generate catalog, wipe + re-embed collection).",
    )
    parser.add_argument(
        "--wait-indexes",
        action="store_true",
        help="After creating indexes, poll Atlas until they reach READY.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final summary as JSON (for machine consumption).",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.only and args.skip:
        parser.error("--only and --skip are mutually exclusive.")

    selected = STAGES
    if args.only:
        selected = [s.strip() for s in args.only.split(",") if s.strip()]
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        selected = [s for s in STAGES if s not in skip]

    print(f"[generate] stages: {selected}, force={args.force}")
    started = time.time()
    try:
        summary = run_all(stages=selected, force=args.force, wait_indexes=args.wait_indexes)
    except Exception as exc:
        print(f"[generate] FAILED: {exc}", file=sys.stderr)
        return 1

    summary["total_duration_s"] = round(time.time() - started, 2)
    if args.json:
        # Single line JSON on a dedicated marker so subprocess wrappers can find it
        # even if earlier progress lines happen to start with `{`.
        print("[generate-json]" + json.dumps(summary, default=str))
    else:
        print(f"[generate] done in {summary['total_duration_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

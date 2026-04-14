# Advanced Vector Search + Multi-Agent AI Workshop

An advanced hands-on workshop: build AI-powered e-commerce search with **MongoDB Atlas**, **Voyage AI**, and a **multi-agent layer** powered by **Amazon Bedrock (Claude 4.5 Haiku)** and **MongoDB Change Streams**.

## What You'll Build

### Part 1: Search Foundation
- Load a 1,000-product catalog and generate embeddings with Voyage AI
- Build four search modes: **Semantic**, **Text**, **Hybrid**, and **Hybrid + Rerank**
- Run a FastAPI server with a browser-based product search UI

### Part 2: Multi-Agent AI
- **Catalog Watcher Agent** — uses MongoDB Change Streams to detect new product inserts in real time
- **Product Enrichment Agent** — calls Claude 4.5 Haiku to generate descriptions, categories, tags, and embeddings for bare products
- **Shopping Assistant Agent** — conversational AI with tool-use that searches, compares, and recommends products

## Tech Stack

| Technology | Role |
|---|---|
| MongoDB Atlas | Document database, Vector Search, Atlas Search, Change Streams |
| Voyage AI | Embeddings (`voyage-4-large`), Reranking (`rerank-2.5`) |
| Amazon Bedrock | Claude 4.5 Haiku — enrichment + shopping assistant (tool use) |
| FastAPI | Python API server |

## Quick Start

```bash
# Clone
git clone https://github.com/ycyeo-mongodb/Advanced_GenAI_MongoDB_Workshop.git
cd Advanced_GenAI_MongoDB_Workshop

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your MongoDB URI, Voyage AI key, and Bedrock API URL

# Run the numbered scripts in order
python 01_load_and_embed.py     # Load products + generate embeddings
python 02_create_indexes.py     # Create search indexes
python 03_semantic_search.py    # Test semantic search
python 04_hybrid_search.py      # Test hybrid search
python 05_reranking.py          # Test reranking (bonus)

# Start the search app
uvicorn app:app --reload        # Open http://localhost:8000

# --- Part 2: Agents ---

# Terminal 1: Start the Catalog Watcher
python 06_catalog_watcher.py

# Terminal 2: Insert a bare product (triggers enrichment)
python 07_add_bare_product.py

# Terminal 3: Start the Shopping Assistant
python 08_shopping_assistant.py
```

## Scripts

| Script | Description |
|---|---|
| `01_load_and_embed.py` | Load 1,000 products and generate Voyage AI embeddings |
| `02_create_indexes.py` | Create Atlas Vector Search and Atlas Search indexes |
| `03_semantic_search.py` | Test semantic (vector) search |
| `04_hybrid_search.py` | Test hybrid search with RRF fusion |
| `05_reranking.py` | Test reranking with Voyage AI rerank-2.5 |
| `06_catalog_watcher.py` | Change Streams agent — watches for new products |
| `07_add_bare_product.py` | Insert a bare product to trigger enrichment |
| `08_shopping_assistant.py` | Conversational shopping agent with tool use |
| `app.py` | Full FastAPI application (all search modes) |
| `app_starter.py` | Skeleton app for developer track |

## Prerequisites

- Python 3.10+
- MongoDB Atlas account (free tier works)
- Voyage AI API key (via Atlas → Services → AI Models)
- Bedrock API URL (provided by instructor, needed for Part 2)

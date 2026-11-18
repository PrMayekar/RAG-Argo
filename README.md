# 🌊 RAG-Argo: Ocean Intelligence

**Explore Argo float data with ten analytics algorithms and ask questions about it in plain English.**

RAG-Argo turns raw [Argo](https://argo.ucsd.edu/) NetCDF profile files into a queryable ocean data platform. An ETL pipeline loads the data into PostgreSQL, embeds each float profile into ChromaDB, and serves everything through a Django dashboard with interactive Plotly charts and a Retrieval-Augmented Generation (RAG) chat powered by Groq's LLaMA-3.3 70B.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Django](https://img.shields.io/badge/Django-4.2%2B-092E20)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-SQLAlchemy-336791)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)
![LLM](https://img.shields.io/badge/LLM-Groq%20LLaMA--3.3%2070B-purple)

---

## ✨ Features

- **ETL pipeline**: reads Argo NetCDF (`*_prof.nc`) files, validates coordinates, cleans QC flags, and loads floats, profiles and measurements into PostgreSQL. Re-runs are safe and skip data that already exists.
- **RAG chat**: each profile is summarized into a text chunk (location, date, surface and deep readings, column ranges), embedded locally with `all-MiniLM-L6-v2`, and stored in ChromaDB. Questions retrieve the top 5 matching profiles, and the LLM answers using only that data.
- **10 analytics views** in a single-page dashboard (see below).
- **Demo mode**: no database? The dashboard generates synthetic Argo-like data so every feature still works.
- **Graceful chat fallback**: without a `GROQ_API_KEY`, the chat returns canned demo answers instead of failing.
- **Test scripts** for the database, vector store and LLM layers.

## 📊 Analytics Algorithms

| # | View | What it does |
|---|------|--------------|
| 01 | Trajectory Prediction | Linear regression on lat/lon over time, with past and future projections |
| 02 | T-S Diagram | Temperature vs. pressure scatter, coloured by cycle |
| 03 | Temperature Profile | Linear fit of temperature across depth |
| 04 | K-Means Clustering | Geographic clustering of positions (adjustable *k*) |
| 05 | PCA Projection | Dimensionality reduction of float positions |
| 06 | Density Heatmap | Observation density map |
| 07 | kNN Nearest Neighbors | Finds the closest observations to a query lat/lon |
| 08 | Float Summary Table | Aggregated statistics per cycle |
| 09 | Temperature Statistics | Distribution summary with histogram |
| 10 | Top Profile Charts | Temperature profiles for the first 3 cycles |

## 🏗️ Architecture

```
 NetCDF files ──► transform ──► load ──► PostgreSQL ──► embed ──► ChromaDB
 (NetCDF-Files/)   (xarray)    (SQLAlchemy)  floats /               (vectors)
                                             profiles /                │
                                             measurements              │
                                                  │                    ▼
                                                  ▼            Top-5 retrieval
                                         Django dashboard ◄──── + Groq LLaMA-3.3
                                      (10 algorithms + chat)
```

**Database schema**

| Table | Key fields |
|-------|-----------|
| `argo_floats` | `wmo_id` (unique), `dac`, `institution`, `platform_type` |
| `profiles` | `float_id`, `cycle_number`, `profile_date`, `latitude`, `longitude`, `direction`, `profile_pres_qc`, `source_file` |
| `measurements` | `profile_id`, `depth_level`, `pressure`, `temperature`, `salinity`, plus a QC flag for each |

Only measurements with good or probably-good QC flags (`1`, `2`) are used when building embedding chunks.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL running locally or remotely
- A [Groq API key](https://console.groq.com/) (optional; needed only for real chat answers)

### 1. Install

```bash
git clone <your-repo-url>
cd RAG-Argo-main
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# PostgreSQL
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=5432
DB_NAME=argo_db

# Groq LLM
GROQ_API_KEY=your_groq_api_key_here

# ChromaDB path (default: ./chroma_db)
CHROMA_PATH=./chroma_db
```

Create the empty database first, e.g. `createdb argo_db`.

### 3. Build the data pipeline

Put your Argo profile files in `NetCDF-Files/` (ten sample files for 1–10 Jan 2026 are included), then run:

```bash
python main.py
```

This will:

1. Create the `argo_floats`, `profiles` and `measurements` tables.
2. Transform and load every `*.nc` file.
3. Embed all profiles into ChromaDB (the ~90 MB embedding model downloads on first run).

### 4. Launch the dashboard

```bash
python manage.py runserver
```

Open **http://localhost:8000**.

> The dashboard works even without steps 3 and 4's data: if the database isn't reachable it falls back to synthetic demo data.

## 💬 Using the Chat

Ask the Ocean AI questions grounded in your data, for example:

- *"What was the surface temperature of float 5907176?"*
- *"Which profiles were recorded on 2026-01-03?"*
- *"What is the salinity range for cycle 12?"*

You can also chat from the terminal:

```bash
python query.py
```

## 🔌 API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | Dashboard page |
| GET | `/api/floats/` | List available float WMO IDs |
| POST | `/api/graph/` | Run an algorithm. Body: `algorithm`, `float_wmo`, `k_clusters`, `lat_input`, `lon_input` |
| POST | `/api/ask/` | RAG chat. Body: `question` |

Algorithm IDs: `trajectory`, `ts_diagram`, `temp_profile`, `kmeans`, `pca`, `heatmap`, `knn`, `summary`, `stats`, `profiles`.

## 🧪 Testing

The tests are plain scripts that need a populated database and ChromaDB index (run `python main.py` first):

```bash
python run_tests.py      # runs test_etl.py, test_rag.py and test_llm.py
```

- `test_etl.py`: tables are populated, no orphaned rows, values are in valid ranges
- `test_rag.py`: the vector store is populated and float-specific queries return relevant results
- `test_llm.py`: the Groq connection and answer generation

## 📁 Project Structure

```
RAG-Argo-main/
├── NetCDF-Files/          # Input Argo profile files (*.nc)
├── argo_site/             # Django project config (settings, urls, wsgi/asgi)
├── dashboard/             # Django app: views, URLs, API endpoints
├── templates/dashboard/
│   └── index.html         # Single-page dashboard UI (Plotly)
├── db/
│   ├── connection.py      # SQLAlchemy engine and session
│   └── models.py          # ArgoFloat, Profile, Measurement
├── etl/
│   ├── transform.py       # NetCDF → clean dicts (xarray)
│   ├── load.py            # Idempotent PostgreSQL loader
│   └── embed.py           # Profile chunking + ChromaDB indexing
├── main.py                # Runs the full ETL + embedding pipeline
├── query.py               # Terminal RAG chat
├── app.py                 # Standalone Streamlit version of the dashboard
├── abc.py                 # Standalone matplotlib analytics script
├── test_*.py, run_tests.py
├── requirements.txt
└── .env.example
```

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Web | Django, Plotly.js |
| Data | PostgreSQL, SQLAlchemy, pandas, NumPy, xarray, netCDF4 |
| ML | scikit-learn (LinearRegression, KMeans, PCA, NearestNeighbors) |
| RAG | ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`), Groq (`llama-3.3-70b-versatile`) |

## 📝 Notes

- `app.py` is an earlier Streamlit prototype of the dashboard. It needs `pip install streamlit` and has hard-coded database credentials you must edit.
- `abc.py` is an exploratory script with a hard-coded local folder path; update it before running.
- `settings.py` ships with `DEBUG = True`, `ALLOWED_HOSTS = ['*']` and a placeholder `SECRET_KEY`. Change these before any real deployment.

## 🙏 Acknowledgements

Data from the [Argo Program](https://argo.ucsd.edu/), which is part of the Global Ocean Observing System.
# ACM Trend Explorer UI

This directory contains a standalone Streamlit interface for exploring the 
custom ACM CCS hierarchy against the arXiv trend dataset (`arxiv_trends.csv.bz2`)
that lives under `llmdap/profiler`.

## Quick start

```bash
python3 -m venv .venv-ui
source .venv-ui/bin/activate
pip install --upgrade pip
pip install -r ui/trend_explorer/requirements.txt
PYTHONPATH=$(pwd) streamlit run ui/trend_explorer/app.py
```

The `PYTHONPATH` flag lets the UI import the existing `llmdap` package and
re-use the hierarchy and dataset without modifying the core project tree.

#!/usr/bin/env bash
# Daily refresh: data -> tables -> lines -> predictions (updates index.html and picks.json)
set -e
python fetch_data.py
python prep.py data/raw/mbb
python players.py          # rebuilds data/missing.parquet (only needed if you re-tune)
python fetch_lines.py || echo "no lines fetched - type them into the dashboard instead"
python predict.py data/raw/mbb 3

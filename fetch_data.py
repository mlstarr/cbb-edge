"""Download hoopR parquet files (team box, player box, schedules). Past seasons are cached; the current season is refreshed each run.
UNTESTED here (my sandbox can't reach raw.githubusercontent.com); the same files were verified from a git clone."""
import os, sys, requests, datetime as dt
B = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb"
cur = dt.date.today().year + (1 if dt.date.today().month >= 8 else 0)
for y in range(2018, cur + 1):
    for sub, name in [("team_box", f"team_box_{y}.parquet"), ("player_box", f"player_box_{y}.parquet"), ("schedules", f"mbb_schedule_{y}.parquet")]:
        path = f"data/raw/mbb/{sub}/parquet/{name}"
        if os.path.exists(path) and y < cur: continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(f"{B}/{sub}/parquet/{name}", timeout=120); r.raise_for_status(); open(path, "wb").write(r.content); print("got", path)

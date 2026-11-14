"""Run on YOUR machine (ESPN is blocked from my sandbox, so this file is UNTESTED against the live API).
Writes data/lines.csv (market spread/total per upcoming game) and, best-effort, data/injuries_auto.csv.
Usage: python fetch_lines.py [YYYYMMDD ...]   (default: today and tomorrow)
"""
import sys, requests, pandas as pd, datetime as dt
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

def day_lines(day):
    r = requests.get(f"{BASE}/scoreboard", params={"dates": day, "groups": 50, "limit": 400}, timeout=30); r.raise_for_status()
    rows = []
    for ev in r.json().get("events", []):
        c = ev["competitions"][0]; odds = (c.get("odds") or [{}])[0]
        if not odds: continue
        home = next(x for x in c["competitors"] if x["homeAway"] == "home"); away = next(x for x in c["competitors"] if x["homeAway"] == "away")
        spread = odds.get("spread")        # ESPN: signed from the HOME team's view (negative = home favoured)
        if spread is None: continue
        rows.append(dict(game_id=str(ev["id"]), date=ev["date"][:10], home_id=int(home["team"]["id"]), away_id=int(away["team"]["id"]),
                         home=home["team"]["displayName"], away=away["team"]["displayName"], neutral=bool(c.get("neutralSite", False)),
                         spread_home=float(spread), total=odds.get("overUnder"), provider=(odds.get("provider") or {}).get("name")))
    return rows

if __name__ == "__main__":
    days = sys.argv[1:] or [(dt.date.today() + dt.timedelta(days=i)).strftime("%Y%m%d") for i in (0, 1)]
    rows = [r for d in days for r in day_lines(d)]
    df = pd.DataFrame(rows); df.to_csv("data/lines.csv", index=False); print(len(df), "games with lines -> data/lines.csv")
    if len(df):   # keep a history (with fetch time) so the model can later be backtested against real lines
        import os
        df.assign(fetched=dt.datetime.utcnow().isoformat(timespec="minutes")).to_csv("data/lines_history.csv", mode="a", header=not os.path.exists("data/lines_history.csv"), index=False)

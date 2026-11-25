"""Best-effort injury feed from ESPN's public game-summary endpoint (no key). Writes data/injuries_auto.csv and a coverage probe
(data/injury_probe.json) so we can see whether ESPN publishes college injuries at all. Written without access to ESPN from the build
sandbox: the response shape is assumed from ESPN's other sports and every parse step is defensive."""
import sys, json, requests, pandas as pd, datetime as dt
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

def events(day):
    r = requests.get(f"{BASE}/scoreboard", params={"dates": day, "groups": 50, "limit": 400}, timeout=30); r.raise_for_status()
    return [e["id"] for e in r.json().get("events", [])]

def injuries_for(eid):
    r = requests.get(f"{BASE}/summary", params={"event": eid}, timeout=30); r.raise_for_status()
    out = []
    for t in r.json().get("injuries", []) or []:
        team = (t.get("team") or {}).get("displayName")
        for i in t.get("injuries", []) or []:
            name = (i.get("athlete") or {}).get("displayName"); status = i.get("status") or (i.get("type") or {}).get("description") or ""
            if team and name: out.append(dict(team=team, player=name, status=str(status).lower().replace("out for season", "out").strip(), detail=((i.get("details") or {}).get("type") or "")))
    return out

if __name__ == "__main__":
    days = sys.argv[1:] or [(dt.date.today() + dt.timedelta(days=i)).strftime("%Y%m%d") for i in (0, 1)]
    rows, n_ev, n_with, errs = [], 0, 0, 0
    for d in days:
        try: ids = events(d)
        except Exception as e: print("scoreboard failed", d, e); continue
        for eid in ids:
            n_ev += 1
            try:
                r = injuries_for(eid); n_with += bool(r); rows += r
            except Exception: errs += 1
    df = pd.DataFrame(rows, columns=["team", "player", "status", "detail"]).drop_duplicates(["team", "player"])
    df.to_csv("data/injuries_auto.csv", index=False)
    json.dump(dict(run=dt.datetime.utcnow().isoformat(timespec="minutes"), events=n_ev, events_with_injuries=n_with, entries=len(df), errors=errs, sample=df.head(5).to_dict("records")), open("data/injury_probe.json", "w"), indent=1)
    print(f"{n_ev} games checked, {n_with} with injury data, {len(df)} players -> data/injuries_auto.csv")

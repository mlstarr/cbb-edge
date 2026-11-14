# CBB edge finder
Daily GitHub Action: downloads hoopR box scores, fetches market lines from ESPN's public scoreboard feed (no API key),
refits the odds-ratio model, rewrites `index.html` (serve with GitHub Pages) and logs lines + predictions for future backtests.
Optional: add `data/injuries.csv` (team,player,status) to override the automatic absence flags. Run "Actions > daily-predictions > Run workflow" to test.

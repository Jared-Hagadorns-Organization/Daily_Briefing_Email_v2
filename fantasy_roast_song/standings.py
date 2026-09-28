"""Pull Wizards League standings + a week's results from Sleeper for the roast song.

Usage:
    python fantasy_roast_song/standings.py            # last fully scored week
    python fantasy_roast_song/standings.py --week 3   # a specific week (may be live)

Standings use Sleeper's official records, which include the league-median game
(league_average_match), so each scored week counts as two games.
"""

import argparse

import httpx

LEAGUE_ID = "1312062559569862656"
API = "https://api.sleeper.app/v1"

# Sleeper username -> league nickname used in the lyrics.
NICKNAMES = {
    "Jonsp0311": "Jon",
    "MAGAritaville": "Jarrod",
    "Leeroy704": "Leeroy",
    "lukeraynor": "Luke",
    "Benk17": "Ben",
    "CornstarchKing": "Brandon",
    "SheriffKyle": "Kyle",
    "AmcShortSqueeze": "Pollitto",
    "MoneyTeamFC": "Craig",
    "SouthNeck": "Matt",
    "jaredhagadorn": "Jared",
    "Blousesss": "Andy",
}


def get(client: httpx.Client, path: str):
    resp = client.get(f"{API}{path}")
    resp.raise_for_status()
    return resp.json()


def points(settings: dict, key: str) -> float:
    return settings.get(key, 0) + settings.get(f"{key}_decimal", 0) / 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int)
    args = parser.parse_args()

    with httpx.Client(timeout=60) as client:
        league = get(client, f"/league/{LEAGUE_ID}")
        users = {u["user_id"]: u for u in get(client, f"/league/{LEAGUE_ID}/users")}
        rosters = get(client, f"/league/{LEAGUE_ID}/rosters")
        week = args.week or league["settings"]["last_scored_leg"]
        matchups = get(client, f"/league/{LEAGUE_ID}/matchups/{week}")
        players = get(client, "/players/nfl")

    def who(roster: dict) -> str:
        user = users.get(roster["owner_id"], {})
        name = user.get("display_name", "?")
        team = (user.get("metadata") or {}).get("team_name") or name
        return f"{NICKNAMES.get(name, name)} ({team.strip()})"

    def pname(pid: str) -> str:
        return players.get(pid, {}).get("full_name", pid)

    by_roster = {r["roster_id"]: r for r in rosters}
    ranked = sorted(
        rosters,
        key=lambda r: (r["settings"]["wins"], points(r["settings"], "fpts")),
    )

    print(f"{league['name']} — standings through week {league['settings']['last_scored_leg']}")
    print("Worst to best:")
    for rank, r in zip(range(len(ranked), 0, -1), ranked):
        s = r["settings"]
        print(
            f"  #{rank:>2} {who(r):<34} {s['wins']}-{s['losses']}"
            f"  PF {points(s, 'fpts'):7.2f}  PA {points(s, 'fpts_against'):7.2f}"
        )

    print(f"\nWeek {week} results:")
    games: dict[int, list] = {}
    for m in matchups:
        games.setdefault(m["matchup_id"], []).append(m)
    for pair in games.values():
        pair.sort(key=lambda m: -(m["points"] or 0))
        parts = []
        for m in pair:
            sp, st = m["starters_points"], m["starters"]
            hi = max(range(len(st)), key=sp.__getitem__)
            lo = min(range(len(st)), key=sp.__getitem__)
            parts.append(
                f"{who(by_roster[m['roster_id']])} {m['points']:.2f}"
                f" [MVP {pname(st[hi])} {sp[hi]} | dud {pname(st[lo])} {sp[lo]}]"
            )
        print("  " + "  def.  ".join(parts))


if __name__ == "__main__":
    main()

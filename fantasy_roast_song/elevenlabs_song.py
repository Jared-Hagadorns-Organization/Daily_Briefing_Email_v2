"""Generate the full roast song (vocals + beat) with the ElevenLabs Music API.

Reads the lyrics from README.md, turns each section into a composition-plan chunk
with its own duration, and saves the result as roast_song.mp3.

Usage:
    export ELEVENLABS_API_KEY=...        # set in the environment, never commit it
    python fantasy_roast_song/elevenlabs_song.py --dry-run   # print the plan only
    python fantasy_roast_song/elevenlabs_song.py             # generate the song
"""

import argparse
import json
import os
import re
from pathlib import Path

import httpx

HERE = Path(__file__).parent
URL = "https://api.elevenlabs.io/v1/music"
MS_PER_CHAR = 70  # roughly the pace of a half-sung, half-rapped delivery

STYLES = [
    "retro funk", "80s boogie", "feel-good party groove", "104 bpm", "slap bass",
    "electric piano stabs", "synth brass hits", "handclaps", "male vocal",
    "smooth warm baritone", "playful half-sung half-rapped delivery", "comedic",
    "call-and-response backing vocals",
]
NEGATIVE = ["trap", "heavy 808", "autotune", "dark", "aggressive", "metal", "EDM drop"]


def load_sections() -> list[tuple[str, list[str]]]:
    readme = (HERE / "README.md").read_text()
    body = readme.split("## Lyrics", 1)[1].split("\n---", 1)[0]
    sections: list[tuple[str, list[str]]] = []
    for line in body.strip().splitlines():
        header = re.match(r"\*\*\[(.+?)\]\*\*", line)
        if header:
            sections.append((header.group(1), []))
        elif line.strip() and sections:
            sections[-1][1].append(line.replace("*", "").strip())
    # Bare "[Hook]" repeats have no lines in the README; reuse the first hook's.
    hook = next(lines for name, lines in sections if name == "Hook" and lines)
    return [(name, lines or hook) for name, lines in sections]


def build_plan() -> dict:
    chunks = []
    for name, lines in load_sections():
        spoken = "spoken" in name
        label = "Verse" if name.startswith("#") else re.sub(r" — .*", "", name)
        chunks.append({
            "text": f"[{label}]\n" + "\n".join(lines),
            "duration_ms": int(min(120_000, max(4_000, len("".join(lines)) * MS_PER_CHAR))),
            "positive_styles": STYLES + (["spoken word over the groove"] if spoken else []),
            "negative_styles": NEGATIVE,
        })
    return {"chunks": chunks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="music_v2_5")
    parser.add_argument("--out", default=str(HERE / "roast_song.mp3"))
    args = parser.parse_args()

    plan = build_plan()
    total = sum(c["duration_ms"] for c in plan["chunks"]) / 1000
    print(f"{len(plan['chunks'])} chunks, {total:.0f}s total")
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    resp = httpx.post(
        URL,
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        json={"composition_plan": plan, "model_id": args.model},
        timeout=600,
    )
    if resp.status_code != 200:
        raise SystemExit(f"ElevenLabs error {resp.status_code}: {resp.text[:500]}")
    Path(args.out).write_bytes(resp.content)
    print(f"Saved {args.out} ({len(resp.content) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

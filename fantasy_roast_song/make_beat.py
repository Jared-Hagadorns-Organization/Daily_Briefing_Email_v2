"""Generate the original backing track for the roast song.

Writes two files next to this script:
    beat.mid  - multitrack MIDI (drums, slap bass, e-piano, synth brass) to open in
                GarageBand / FL / Ableton and swap in real instrument sounds
    beat.wav  - a quick synthesized preview of the same arrangement

The groove is an original retro-funk pattern (104 BPM, F major, I-vi-ii-V verses,
IV-V-iii-vi hook). The arrangement is laid out bar-for-bar against the lyric
sections in README.md, so a vocal take lines up without editing.

Usage:
    pip install numpy mido
    python fantasy_roast_song/make_beat.py [--preview]   # --preview renders 32 bars only
"""

import argparse
import wave
from pathlib import Path

import mido
import numpy as np

BPM = 104
SR = 22050
BEAT = 60 / BPM
OUT = Path(__file__).parent

# (section name, bars). Verse lengths follow the lyric line counts: one line per bar.
ARRANGEMENT = [
    ("intro", 4), ("hook", 8),
    ("verse", 6),  # 12 Jon
    ("verse", 6),  # 11 Leeroy
    ("hook", 8),
    ("verse", 8),  # 10 Luke
    ("verse", 8),  # 9 Jarrod
    ("verse", 6),  # 8 Kyle
    ("hook", 8),
    ("verse", 8),  # 7 Ben
    ("verse", 8),  # 6 Andy
    ("verse", 6),  # 5 Pollitto
    ("verse", 6),  # 4 Matt
    ("verse", 12),  # 3 Brandon
    ("hook", 8),
    ("verse", 6),  # 2 Craig
    ("verse", 8),  # 1 Jared
    ("hook", 8), ("outro", 4),
]

# Chords as (root MIDI note, chord tones as semitone offsets). F major.
FMAJ7, DM7, GM7, C7 = (41, [0, 4, 7, 11]), (38, [0, 3, 7, 10]), (43, [0, 3, 7, 10]), (36, [0, 4, 7, 10])
BBMAJ7, AM7 = (46, [0, 4, 7, 11]), (45, [0, 3, 7, 10])
VERSE_CHORDS = [FMAJ7, DM7, GM7, C7]
HOOK_CHORDS = [BBMAJ7, C7, AM7, DM7]

# 16th-step patterns (16 steps per bar).
KICK = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
SNARE = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
HAT = [1, 1, 1, 1] * 4
# Bass: (step, chord-tone index or 'oct' pop, length in steps)
BASS = [(0, 0, 3), (3, "oct", 1), (6, 0, 2), (8, 2, 2), (10, 0, 1), (11, "oct", 1), (14, 1, 2)]
KEYS_STEPS = [2, 7, 10, 15]  # off-beat e-piano stabs
SWING = 0.12  # delay of every other 16th, as a fraction of a 16th


def step_time(bar: int, step: int) -> float:
    sixteenth = BEAT / 4
    return bar * 4 * BEAT + step * sixteenth + (SWING * sixteenth if step % 2 else 0)


def build_events(arrangement):
    """Return a list of (instrument, start_sec, dur_sec, midi_note, velocity)."""
    ev = []
    bar = 0
    for section, bars in arrangement:
        chords = HOOK_CHORDS if section == "hook" else VERSE_CHORDS
        for b in range(bars):
            root, tones = chords[b % 4]
            full = section in ("hook", "verse")
            last_bar = b == bars - 1
            # Drums
            for s in range(16):
                t = step_time(bar, s)
                if KICK[s] and (full or s == 0):
                    ev.append(("kick", t, 0.3, 36, 110))
                if SNARE[s] and full and not (s == 15 and not last_bar):
                    ev.append(("snare", t, 0.2, 38, 105 if s != 15 else 70))
                    if section == "hook" and s in (4, 12):
                        ev.append(("clap", t, 0.2, 39, 100))
                if HAT[s]:
                    ev.append(("hat", t, 0.05, 42, 80 if s % 4 == 2 else 50))
            # Bass (enters after the intro's first two bars)
            if full or (section == "intro" and b >= 2) or section == "outro":
                for s, tone, length in BASS:
                    note = root + 12 if tone == "oct" else root + tones[tone]
                    ev.append(("bass", step_time(bar, s), length * BEAT / 4 * 0.9, note, 100))
            # E-piano stabs
            for s in KEYS_STEPS:
                for tone in tones:
                    ev.append(("keys", step_time(bar, s), BEAT / 4 * 1.5, root + 24 + tone, 70))
            # Synth brass: a rising hit on each verse's first bar (the rank reveal),
            # and a call-and-response riff through the hook.
            if section == "verse" and b == 0:
                for i, off in enumerate([0, 4, 7]):
                    ev.append(("brass", step_time(bar, i * 2), BEAT / 2, root + 36 + off, 95))
            if section == "hook" and b % 2 == 1:
                for s, off in [(8, 7), (10, 9), (12, 12), (14, 9)]:
                    ev.append(("brass", step_time(bar, s), BEAT / 4 * 1.6, root + 36 + off, 90))
            bar += 1
    return ev, bar


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------
GM = {"bass": (0, 36), "keys": (1, 4), "brass": (2, 62)}  # channel, program


def write_midi(events, path: Path) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    tracks = {}
    for name in ["drums", "bass", "keys", "brass"]:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if name == "drums":
            tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
        else:
            ch, prog = GM[name]
            tr.append(mido.Message("program_change", channel=ch, program=prog, time=0))
        tracks[name] = (tr, [])
        mid.tracks.append(tr)
    for inst, start, dur, note, vel in events:
        name = "drums" if inst in ("kick", "snare", "clap", "hat") else inst
        ch = 9 if name == "drums" else GM[name][0]
        on = round(start / BEAT * 480)
        off = on + max(1, round(dur / BEAT * 480))
        msgs = tracks[name][1]
        msgs.append((on, 1, mido.Message("note_on", channel=ch, note=note, velocity=vel)))
        msgs.append((off, 0, mido.Message("note_off", channel=ch, note=note, velocity=0)))
    for tr, msgs in tracks.values():
        now = 0
        for tick, _, msg in sorted(msgs, key=lambda m: (m[0], m[1])):
            tr.append(msg.copy(time=tick - now))
            now = tick
    mid.save(path)


# ---------------------------------------------------------------------------
# Preview synthesis
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)


def hz(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)


def lowpass(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        y[i] = acc
    return y


def voice(inst: str, dur: float, note: int) -> np.ndarray:
    if inst == "kick":
        n = int(0.35 * SR)
        t = np.arange(n) / SR
        freq = 50 + 90 * np.exp(-t / 0.03)
        return np.sin(2 * np.pi * np.cumsum(freq) / SR) * env(n, 0.001, 0.12)
    if inst == "snare":
        n = int(0.2 * SR)
        t = np.arange(n) / SR
        return 0.6 * lowpass(rng.standard_normal(n), 0.5) * env(n, 0.001, 0.05) + 0.4 * np.sin(2 * np.pi * 190 * t) * env(n, 0.001, 0.04)
    if inst == "clap":
        n = int(0.18 * SR)
        noise = rng.standard_normal(n)
        e = sum(np.roll(env(n, 0.001, 0.012), int(k * 0.011 * SR)) for k in range(3)) + env(n, 0.001, 0.06)
        return 0.5 * noise * e
    if inst == "hat":
        n = int(0.05 * SR)
        noise = rng.standard_normal(n)
        return 0.35 * np.diff(noise, prepend=0) * env(n, 0.0005, 0.012)
    n = int((dur + 0.08) * SR)
    t = np.arange(n) / SR
    f = hz(note)
    if inst == "bass":
        saw = 2 * ((t * f) % 1) - 1
        pluck = lowpass(saw, 0.08) + 0.5 * lowpass(saw, 0.35) * env(n, 0.001, 0.03)
        gate = (t < dur).astype(float)
        return 0.9 * pluck * env(n, 0.002, 0.25) * np.maximum(gate, np.exp(-(t - dur).clip(0) / 0.02))
    if inst == "keys":
        tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) * env(n, 0.001, 0.05)
        return 0.18 * tone * env(n, 0.002, 0.18)
    if inst == "brass":
        saw = sum(2 * ((t * f * d) % 1) - 1 for d in (0.996, 1.0, 1.004)) / 3
        return 0.3 * lowpass(saw, 0.25) * env(n, 0.02, dur * 0.9 + 0.05)
    raise ValueError(inst)


LEVEL = {"kick": 0.9, "snare": 0.55, "clap": 0.45, "hat": 0.22, "bass": 0.55, "keys": 1.0, "brass": 0.6}


def render(events, total_bars: int, path: Path) -> None:
    length = int((total_bars * 4 * BEAT + 1.0) * SR)
    mix = np.zeros(length)
    cache = {}
    for inst, start, dur, note, vel in events:
        key = (inst, round(dur, 3), note)
        if key not in cache:
            cache[key] = voice(inst, dur, note)
        sig = cache[key] * LEVEL[inst] * vel / 127
        i = int(start * SR)
        seg = sig[: max(0, length - i)]
        mix[i : i + len(seg)] += seg
    mix = np.tanh(1.4 * mix)  # gentle glue/saturation
    mix /= np.abs(mix).max() / 0.9
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="render only intro + hook + 2 verses")
    args = parser.parse_args()
    arrangement = ARRANGEMENT[:5] if args.preview else ARRANGEMENT
    events, bars = build_events(arrangement)
    write_midi(events, OUT / "beat.mid")
    wav = OUT / ("beat_preview.wav" if args.preview else "beat.wav")
    render(events, bars, wav)
    print(f"{bars} bars at {BPM} BPM = {bars * 4 * BEAT:.0f}s -> {wav.name}, beat.mid")


if __name__ == "__main__":
    main()

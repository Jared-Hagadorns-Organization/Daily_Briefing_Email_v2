"""Generate a P-Funk-style backing track (v2) for the roast song.

Built from the publicly described traits of the reference song: 109 BPM,
funk/R&B in the Sly Stone / P-Funk / Earth, Wind & Fire lineage, carried by a
driving synth. No audio was sampled or transcribed. The chords, bassline,
riffs and drum pattern are original.

Instruments: Moog-style synth bass with filter envelope, clavinet chicken-scratch,
a sawtooth lead riff on the hooks, EWF-style horn stabs, and a funk kit with snare
ghost notes and open hats.

Writes beat_pfunk.wav and beat_pfunk.mid next to this script.

Usage:
    pip install numpy mido
    python fantasy_roast_song/make_beat_pfunk.py [--preview]
"""

import argparse
import wave
from pathlib import Path

import mido
import numpy as np

from make_beat import ARRANGEMENT, SR, env, hz, lowpass

BPM = 109
BEAT = 60 / BPM
OUT = Path(__file__).parent
SWING = 0.10

# A major funk vamp. (root MIDI note, chord tones as semitone offsets)
A7, D9 = (45, [0, 4, 7, 10]), (38, [0, 4, 10, 14])
DMAJ, E9, CSM7, FSM7 = (38, [0, 4, 7, 11]), (40, [0, 4, 10, 14]), (37, [0, 3, 7, 10]), (42, [0, 3, 7, 10])
VERSE_CHORDS = [A7, A7, D9, D9]
HOOK_CHORDS = [DMAJ, E9, CSM7, FSM7]

KICK = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
SNARE = [0, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 1]  # 2 = backbeat, 1 = ghost
OPEN_HAT = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
# Driving synth bass: (step, semitone offset from root, length in steps)
BASS = [(0, 0, 2), (2, 12, 1), (3, 0, 1), (5, 10, 1), (6, 12, 2), (8, 0, 1),
        (10, 7, 2), (12, 0, 1), (13, 12, 1), (14, 10, 1), (15, 7, 1)]
CLAV = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
# Hook lead riff over two bars: (bar 0/1, step, scale degree as semitones above A4)
LEAD = [(0, 0, 12), (0, 3, 9), (0, 6, 7), (0, 8, 4), (0, 10, 7), (0, 14, 9),
        (1, 0, 12), (1, 2, 14), (1, 4, 12), (1, 7, 9), (1, 10, 7), (1, 12, 4)]


def t_of(bar: int, step: int) -> float:
    s = BEAT / 4
    return bar * 4 * BEAT + step * s + (SWING * s if step % 2 else 0)


def build_events(arrangement):
    ev, bar = [], 0
    for section, bars in arrangement:
        chords = HOOK_CHORDS if section == "hook" else VERSE_CHORDS
        full = section in ("hook", "verse")
        for b in range(bars):
            root, tones = chords[b % 4]
            for s in range(16):
                t = t_of(bar, s)
                if KICK[s] and (full or s == 0):
                    ev.append(("kick", t, 0.3, 36, 115))
                if SNARE[s] and full:
                    ev.append(("snare", t, 0.2, 38, 110 if SNARE[s] == 2 else 40))
                ev.append(("ohat" if OPEN_HAT[s] else "hat", t, 0.1, 46 if OPEN_HAT[s] else 42,
                           70 if OPEN_HAT[s] else 45))
                if CLAV[s] and (full or section == "outro"):
                    for tone in tones[:3]:
                        ev.append(("clav", t, BEAT / 8, root + 24 + tone, 75 if s % 4 else 95))
            if full or b >= 2:
                for s, off, length in BASS:
                    ev.append(("bass", t_of(bar, s), length * BEAT / 4 * 0.85, root + off, 105))
            if section == "verse" and b == 0:  # horn hit on each rank reveal
                for i in range(2):
                    for tone in tones[:3]:
                        ev.append(("horn", t_of(bar, i * 3), BEAT / 2, root + 36 + tone, 105))
            if section in ("verse", "hook") and b % 4 == 3:  # turnaround stab on the "and" of 4
                for tone in tones[:3]:
                    ev.append(("horn", t_of(bar, 14), BEAT / 3, root + 36 + tone, 95))
            if section == "hook":
                for lb, s, deg in LEAD:
                    if lb == b % 2:
                        ev.append(("lead", t_of(bar, s), BEAT / 4 * 1.8, 69 + deg - 12, 100))
            bar += 1
    return ev, bar


# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)


def filtered_saw(f, n, cutoff_env, detune=(1.0,)):
    t = np.arange(n) / SR
    saw = sum(2 * ((t * f * d) % 1) - 1 for d in detune) / len(detune)
    y, acc = np.empty(n), 0.0
    for i in range(n):
        acc += cutoff_env[i] * (saw[i] - acc)
        y[i] = acc
    return y


def voice(inst, dur, note):
    if inst == "kick":
        n = int(0.35 * SR)
        t = np.arange(n) / SR
        freq = 48 + 110 * np.exp(-t / 0.025)
        click = 0.3 * rng.standard_normal(n) * env(n, 0.0005, 0.003)
        return np.sin(2 * np.pi * np.cumsum(freq) / SR) * env(n, 0.001, 0.14) + click
    if inst == "snare":
        n = int(0.22 * SR)
        t = np.arange(n) / SR
        body = np.sin(2 * np.pi * 185 * t) * env(n, 0.001, 0.05)
        return 0.65 * lowpass(rng.standard_normal(n), 0.45) * env(n, 0.001, 0.07) + 0.45 * body
    if inst in ("hat", "ohat"):
        n = int((0.18 if inst == "ohat" else 0.05) * SR)
        return 0.3 * np.diff(rng.standard_normal(n), prepend=0) * env(n, 0.0005, 0.06 if inst == "ohat" else 0.012)
    n = int((dur + 0.06) * SR)
    t = np.arange(n) / SR
    f = hz(note)
    gate = np.where(t < dur, 1.0, np.exp(-(t - dur).clip(0) / 0.015))
    if inst == "bass":  # Moog-ish: square+saw through a snappy filter envelope
        cutoff = 0.03 + 0.25 * np.exp(-t / 0.06)
        sq = np.sign(np.sin(2 * np.pi * f * t))
        y = filtered_saw(f, n, cutoff) + 0.5 * lowpass(sq, 0.05)
        return 0.9 * y * gate
    if inst == "clav":  # bright plucked pulse
        pulse = np.where((t * f) % 1 < 0.25, 1.0, -0.33)
        return 0.22 * lowpass(pulse, 0.8) * env(n, 0.0005, 0.05) * gate
    if inst == "horn":  # brassy swell: detuned saws, filter opens on attack
        cutoff = 0.1 + 0.3 * (1 - np.exp(-t / 0.03)) * np.exp(-t / 0.25)
        return 0.3 * filtered_saw(f, n, cutoff, (0.995, 1.0, 1.005)) * env(n, 0.012, 0.35) * gate
    if inst == "lead":  # the "driving synth": saw lead with a bit of vibrato
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t)
        ph = np.cumsum(f * vib) / SR
        saw = 2 * (ph % 1) - 1
        return 0.3 * lowpass(saw, 0.6) * env(n, 0.005, 0.5) * gate
    raise ValueError(inst)


LEVEL = {"kick": 0.95, "snare": 0.65, "hat": 0.32, "ohat": 0.3, "bass": 0.42,
         "clav": 1.4, "horn": 1.0, "lead": 1.2}


def render(events, bars, path):
    length = int((bars * 4 * BEAT + 1.0) * SR)
    mix, cache = np.zeros(length), {}
    for inst, start, dur, note, vel in events:
        key = (inst, round(dur, 3), note)
        if key not in cache:
            cache[key] = voice(inst, dur, note)
        sig = cache[key] * LEVEL[inst] * vel / 127
        i = int(start * SR)
        seg = sig[: max(0, length - i)]
        mix[i : i + len(seg)] += seg
    mix = np.tanh(1.5 * mix)
    mix /= np.abs(mix).max() / 0.9
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())


GM = {"bass": (0, 38), "clav": (1, 7), "horn": (2, 61), "lead": (3, 81)}
DRUMS = {"kick", "snare", "hat", "ohat"}


def write_midi(events, path):
    mid = mido.MidiFile(ticks_per_beat=480)
    tracks = {}
    for name in ["drums", *GM]:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if name == "drums":
            tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
        else:
            tr.append(mido.Message("program_change", channel=GM[name][0], program=GM[name][1], time=0))
        tracks[name] = (tr, [])
        mid.tracks.append(tr)
    for inst, start, dur, note, vel in events:
        name = "drums" if inst in DRUMS else inst
        ch = 9 if name == "drums" else GM[name][0]
        on = round(start / BEAT * 480)
        off = on + max(1, round(dur / BEAT * 480))
        tracks[name][1].extend([(on, 1, mido.Message("note_on", channel=ch, note=note, velocity=vel)),
                            (off, 0, mido.Message("note_off", channel=ch, note=note, velocity=0))])
    for tr, msgs in tracks.values():
        now = 0
        for tick, _, msg in sorted(msgs, key=lambda m: (m[0], m[1])):
            tr.append(msg.copy(time=tick - now))
            now = tick
    mid.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    arrangement = ARRANGEMENT[:5] if args.preview else ARRANGEMENT
    events, bars = build_events(arrangement)
    write_midi(events, OUT / "beat_pfunk.mid")
    wav = OUT / ("beat_pfunk_preview.wav" if args.preview else "beat_pfunk.wav")
    render(events, bars, wav)
    print(f"{bars} bars at {BPM} BPM = {bars * 4 * BEAT:.0f}s -> {wav.name}, beat_pfunk.mid")


if __name__ == "__main__":
    main()

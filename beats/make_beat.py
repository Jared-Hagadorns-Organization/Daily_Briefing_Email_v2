"""Original royalty-free upbeat acoustic-pop backing track (synthesized from scratch)."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, butter

SR = 44100
BPM = 116
BEAT = 60 / BPM
BAR = 4 * BEAT
rng = np.random.default_rng(7)

def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def pluck(freq, dur, bright=0.5, decay=0.996):
    """Karplus-Strong plucked string."""
    n = int(SR * dur)
    p = max(2, int(SR / freq))
    buf = rng.uniform(-1, 1, p)
    buf = lfilter([bright, 1 - bright], [1], buf)
    out = np.empty(n)
    for i in range(n):
        out[i] = buf[i % p]
        buf[i % p] = decay * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out

def tone(freq, dur, attack=0.005, release=0.15, harmonics=(1, .5, .25)):
    t = np.arange(int(SR * dur)) / SR
    s = sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))
    env = np.minimum(1, t / attack) * np.exp(-t / (dur * 0.9 + 1e-3))
    tail = int(SR * release)
    env[-tail:] *= np.linspace(1, 0, tail)
    return s * env

def kick():
    t = np.arange(int(SR * 0.35)) / SR
    f = 50 + 90 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)

def snare():
    t = np.arange(int(SR * 0.22)) / SR
    noise = lfilter(*butter(2, 1500 / (SR / 2), "high"), rng.uniform(-1, 1, len(t)))
    body = np.sin(2 * np.pi * 190 * t)
    return (0.7 * noise + 0.4 * body) * np.exp(-t * 20)

def hat(open_=False):
    t = np.arange(int(SR * (0.18 if open_ else 0.05))) / SR
    noise = lfilter(*butter(2, 7000 / (SR / 2), "high"), rng.uniform(-1, 1, len(t)))
    return noise * np.exp(-t * (18 if open_ else 70))

def clap():
    t = np.arange(int(SR * 0.2)) / SR
    noise = lfilter(*butter(2, [900 / (SR / 2), 3000 / (SR / 2)], "band"), rng.uniform(-1, 1, len(t)))
    env = np.exp(-t * 25) + 0.6 * np.exp(-np.abs(t - 0.012) * 400)
    return noise * env

# Chord voicings (MIDI). Key: D major. Generic progressions, original arrangement.
CH = {
    "D":  [50, 57, 62, 66, 69],
    "Bm": [47, 54, 59, 62, 66],
    "G":  [43, 50, 55, 59, 62],
    "A":  [45, 52, 57, 61, 64],
    "Em": [40, 47, 52, 55, 59],
    "F#m": [42, 49, 54, 57, 61],
}
ROOT = {"D": 38, "Bm": 35, "G": 31, "A": 33, "Em": 28, "F#m": 30}

VERSE = ["D", "Bm", "G", "A"] * 2
CHORUS = ["G", "A", "D", "Bm", "G", "A", "D", "D"]
BRIDGE = ["Em", "F#m", "G", "A", "Em", "F#m", "G", "A"]
SECTIONS = [  # (chords, drums on?, intensity)
    (["D", "Bm", "G", "A"], "intro", 0.6),
    (VERSE, "verse", 0.8),
    (CHORUS, "chorus", 1.0),
    (VERSE, "verse", 0.85),
    (CHORUS, "chorus", 1.0),
    (BRIDGE, "bridge", 0.9),
    (CHORUS, "chorus", 1.0),
    (CHORUS, "chorus", 1.0),
    (["D", "G", "A", "D"], "outro", 0.7),
]

n_bars = sum(len(c) for c, _, _ in SECTIONS)
total = int(SR * (n_bars * BAR + 3))
gtr, bass, keys, drums = (np.zeros(total) for _ in range(4))

def add(track, sig, t, gain=1.0):
    i = int(t * SR)
    j = min(total, i + len(sig))
    track[i:j] += gain * sig[: j - i]

# Strum pattern: D . D U . U D U  (8th-note grid, 1 = down, -1 = up)
STRUM = [1, 0, 1, -1, 0, -1, 1, -1]
pluck_cache = {}

def strum(chord, t, direction, gain):
    notes = CH[chord] if direction > 0 else CH[chord][::-1][:4]
    for k, m in enumerate(notes):
        key = (m, direction)
        if key not in pluck_cache:
            pluck_cache[key] = pluck(midi_hz(m), BEAT * 2.2, bright=0.35 if direction > 0 else 0.6)
        add(gtr, pluck_cache[key], t + k * 0.011, gain * (1 if direction > 0 else 0.6))

bar_i = 0
for chords, kind, inten in SECTIONS:
    for ci, chord in enumerate(chords):
        t0 = bar_i * BAR
        # Guitar
        for s, d in enumerate(STRUM):
            if d:
                strum(chord, t0 + s * BEAT / 2 + rng.normal(0, 0.004), d, 0.18 * inten)
        # Bass: root-fifth bounce
        if kind != "intro":
            r = ROOT[chord]
            pattern = [(0, r, 0.75), (1.5, r, 0.4), (2, r + 7, 0.75), (3, r + 12, 0.4), (3.5, r + 7, 0.4)]
            for beat, m, dur in pattern:
                add(bass, tone(midi_hz(m), BEAT * dur * 1.6, harmonics=(1, .35, .1)), t0 + beat * BEAT, 0.5 * inten)
        # Keys: offbeat stabs in chorus/bridge
        if kind in ("chorus", "bridge"):
            for beat in (0.5, 1.5, 2.5, 3.5):
                for m in CH[chord][2:]:
                    add(keys, tone(midi_hz(m + 12), BEAT * 0.35, harmonics=(1, .6, .3, .15)), t0 + beat * BEAT, 0.06)
        # Drums
        if kind in ("intro",) and ci < 2:
            for s in range(8):
                add(drums, hat(), t0 + s * BEAT / 2, 0.15)
        elif kind != "outro" or ci < len(chords) - 1:
            for beat in range(4):
                tb = t0 + beat * BEAT
                if beat in (0, 2):
                    add(drums, kick(), tb, 0.9)
                if beat in (1, 3):
                    add(drums, snare(), tb, 0.55)
                    if kind == "chorus":
                        add(drums, clap(), tb, 0.35)
                add(drums, hat(), tb, 0.22)
                add(drums, hat(open_=(kind == "chorus" and beat == 3)), tb + BEAT / 2, 0.16)
            if kind == "chorus" and ci % 2 == 1:
                add(drums, kick(), t0 + 2.5 * BEAT, 0.6)
            if ci == len(chords) - 1 and kind != "outro":  # fill into next section
                for k in range(4):
                    add(drums, snare(), t0 + 3 * BEAT + k * BEAT / 4, 0.3 + 0.1 * k)
        else:
            add(drums, kick(), t0, 0.9)
        bar_i += 1

# Final ring-out chord
end_t = bar_i * BAR
strum("D", end_t, 1, 0.25)
add(drums, kick(), end_t, 0.9)

# Simple stereo mix with a short room reverb
def reverb(x, amt=0.18):
    y = x.copy()
    for d, g in ((0.029, .5), (0.047, .4), (0.071, .3), (0.113, .2)):
        n = int(d * SR)
        y[n:] += g * amt * x[:-n]
    return y

L = 0.9 * reverb(gtr) + bass + 0.6 * reverb(keys) + drums
R = 0.6 * reverb(gtr) + bass + 0.9 * reverb(keys) + drums
mix = np.stack([L, R], axis=1)
mix = np.tanh(mix / np.max(np.abs(mix)) * 1.4)
mix *= 0.9 / np.max(np.abs(mix))
fade = int(SR * 2)
mix[-fade:] *= np.linspace(1, 0, fade)[:, None]

out = __import__("os").path.join(__import__("os").path.dirname(__file__), "sunny_strum_beat_116bpm.wav")
wavfile.write(out, SR, (mix * 32767).astype(np.int16))
print(out, f"{len(mix) / SR:.1f}s")

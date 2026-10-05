"""Overlay a vocal track on a beat/instrumental and save a finished WAV.

Accepts any audio or video format ffmpeg can read (wav, mp3, m4a, mp4, ...).
If the vocal file is a full song (e.g. a Suno download with its own music),
pass --isolate to strip its backing with Demucs first, keeping only the voice.

Usage:
    python mix_vocals.py vocals.wav beat_instrumental.wav
    python mix_vocals.py suno_song.mp3 beat_instrumental.wav --isolate
    python mix_vocals.py vocals.wav beat.wav --offset 4.2 --vocal-db 2 -o final.wav
    python mix_vocals.py sang_over_speakers.m4a beat.wav --align --autotune

Options worth knowing:
    --offset S     start the vocal S seconds into the beat (negative trims the
                   start of the vocal instead)
    --vocal-db D   make the vocal louder (+) or quieter (-) relative to the beat
    --reverb R     0 = dry, 0.25 = default room, 0.5 = big arena
    --no-duck      don't dip the beat slightly while the vocal is singing
    --align        if the beat is audible in your recording (sang along to
                   speakers), strip it out and line the vocal up automatically
    --autotune     snap the vocal to the song's key (classic auto-tuned pop sound)
    --key K        key to tune to, e.g. D, F#, Bb (default D, matching the beat)
    --scale S      major (default), minor, or chromatic
    --retune MS    0 = hard robotic snap (default), 40-80 = more natural
    --polish       make an untrained voice sound produced: auto-tune plus
                   doubled vocals, noise gate, vocal EQ and slapback echo
    --harmony      add a quiet harmony a third above the lead (pairs with --polish)
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

SR = 44100

NOTE_NAMES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6,
              "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10],
          "chromatic": list(range(12))}


def detect_pitch(x: np.ndarray, hop: int = 256, fmin: float = 70, fmax: float = 700) -> np.ndarray:
    """YIN pitch tracker. Returns f0 in Hz per hop (0 = unvoiced)."""
    win, maxlag, minlag = 1024, int(SR / fmin), int(SR / fmax)
    n_frames = max(0, (len(x) - win - maxlag) // hop)
    f0 = np.zeros(n_frames + 1)
    nfft = 1 << int(np.ceil(np.log2(win + maxlag + win)))
    gate = 10 ** (-45 / 20) * (np.abs(x).max() + 1e-9)
    taus = np.arange(maxlag + 1)
    for i in range(n_frames):
        frame = x[i * hop: i * hop + win + maxlag]
        if np.sqrt(np.mean(frame[:win] ** 2)) < gate:
            continue
        r = np.fft.irfft(np.conj(np.fft.rfft(frame[:win], nfft)) * np.fft.rfft(frame, nfft), nfft)[: maxlag + 1]
        csum = np.concatenate([[0.0], np.cumsum(frame ** 2)])
        e = csum[taus + win] - csum[taus]
        d = e[0] + e - 2 * r
        d[0] = 0
        cmnd = d * np.arange(len(d)) / np.maximum(np.cumsum(d), 1e-12)
        cmnd[0] = 1
        below = np.nonzero(cmnd[minlag:] < 0.15)[0]
        if len(below) == 0:
            continue
        tau = minlag + below[0]
        while tau + 1 <= maxlag and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
        if 1 <= tau < maxlag:  # parabolic interpolation for sub-sample accuracy
            a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
            denom = a - 2 * b + c
            tau = tau + (0.5 * (a - c) / denom if denom else 0)
        f0[i] = SR / tau
    # Median filter knocks out stray octave jumps.
    pad = np.pad(f0, 2, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 5), axis=1)
    return np.where((f0 > 0) & (med > 0), med, 0.0)


def target_pitch(f0: np.ndarray, key: str, scale: str, retune_ms: float, hop: int,
                 degree: int = 0, cents: float = 0.0) -> np.ndarray:
    """Snap each detected pitch to the nearest note in the key.

    degree moves the result up/down that many scale steps (2 = a third above,
    for harmonies); cents adds a fixed detune (for doubled vocals).
    """
    root = NOTE_NAMES[key.upper()]
    allowed = np.array([(root + s) % 12 for s in SCALES[scale]])
    out = np.zeros_like(f0)
    voiced = f0 > 0
    midi = 69 + 12 * np.log2(f0[voiced] / 440)
    base = np.floor(midi)
    cands = base[:, None] + np.arange(-2, 3)[None, :]  # nearby semitones
    ok = np.isin(np.mod(cands, 12), allowed)
    dist = np.where(ok, np.abs(cands - midi[:, None]), np.inf)
    snapped = cands[np.arange(len(cands)), np.argmin(dist, axis=1)]
    if degree:
        notes = np.array([n for n in range(128) if n % 12 in allowed])
        pos = np.clip(np.searchsorted(notes, snapped) + degree, 0, len(notes) - 1)
        shift = notes[pos] - snapped
        midi, snapped = midi + shift, snapped + shift
    snapped = snapped + cents / 100
    if retune_ms > 0:  # glide toward the target note instead of jumping
        alpha = 1 - np.exp(-hop / (SR * retune_ms / 1000))
        smooth, prev = np.empty_like(snapped), None
        for i, (s, m) in enumerate(zip(snapped, midi)):
            prev = m if prev is None else prev + alpha * (s - prev)
            smooth[i] = prev
        snapped = smooth
    out[voiced] = 440 * 2 ** ((snapped - 69) / 12)
    return out


def autotune(vox: np.ndarray, key: str, scale: str, retune_ms: float,
             degree: int = 0, cents: float = 0.0, f0: np.ndarray | None = None) -> np.ndarray:
    """Pitch-correct a vocal with TD-PSOLA (keeps the voice's tone, moves the notes)."""
    hop = 256
    x = vox.mean(axis=1).astype(np.float64)
    if f0 is None:
        f0 = detect_pitch(x, hop)
    tgt = target_pitch(f0, key, scale, retune_ms, hop, degree, cents)
    t_frames = np.arange(len(f0)) * hop + 512  # frame centres
    idx = np.arange(len(x))
    f_src = np.interp(idx, t_frames, f0)
    f_tgt = np.interp(idx, t_frames, tgt)
    voiced = np.interp(idx, t_frames, (f0 > 0).astype(float)) > 0.5

    out = np.zeros_like(x)
    wsum = np.zeros_like(x)
    # Unvoiced parts (breaths, "s", "t") pass through untouched.
    out[~voiced] = x[~voiced]
    wsum[~voiced] = 1.0

    edges = np.diff(np.concatenate([[0], voiced.astype(int), [0]]))
    for start, end in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        # Analysis pitch marks: one per period, locked to waveform peaks.
        period = SR / max(f_src[start], 60)
        lo, hi = start, int(min(end, start + period))
        marks = [lo + int(np.argmax(x[lo:hi]))] if hi > lo else []
        while marks:
            period = SR / max(f_src[marks[-1]], 60)
            lo, hi = int(marks[-1] + 0.75 * period), int(min(end, marks[-1] + 1.25 * period))
            if hi <= lo:
                break
            marks.append(lo + int(np.argmax(x[lo:hi])))
        if len(marks) < 2:
            out[start:end] = x[start:end]
            wsum[start:end] = 1.0
            continue
        marks = np.array(marks)
        # Synthesis marks spaced at the corrected period; reuse the nearest grain.
        t = float(marks[0])
        while t < end:
            k = int(np.argmin(np.abs(marks - t)))
            a = marks[k]
            half = int(SR / max(f_src[a], 60))
            g0, g1 = max(0, a - half), min(len(x), a + half)
            grain = x[g0:g1] * np.hanning(g1 - g0)
            s0 = int(t) - (a - g0)
            o0, o1 = max(0, s0), min(len(x), s0 + len(grain))
            if o1 > o0:
                out[o0:o1] += grain[o0 - s0: o1 - s0]
                wsum[o0:o1] += np.hanning(g1 - g0)[o0 - s0: o1 - s0]
            t += SR / max(f_tgt[int(min(t, len(x) - 1))] or f_src[a], 60)
    tuned = out / np.maximum(wsum, 0.5)
    return np.repeat(tuned[:, None], 2, axis=1).astype(np.float32)


def load(path: Path, tmp: Path) -> np.ndarray:
    """Decode any media file to float32 stereo at 44.1 kHz via ffmpeg."""
    wav = tmp / f"{path.stem}_decoded.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(wav)],
        check=True,
    )
    data, _ = sf.read(wav, dtype="float32", always_2d=True)
    return data


def isolate_vocals(path: Path, tmp: Path) -> tuple[Path, Path]:
    """Run Demucs and return the paths of the (vocals, music) stems."""
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-o", str(tmp / "sep"), str(path)],
        check=True,
    )
    stems = tmp / "sep" / "htdemucs" / path.stem
    return stems / "vocals.wav", stems / "no_vocals.wav"


def find_offset(music: np.ndarray, beat: np.ndarray) -> tuple[float, float]:
    """Find where the beat starts inside a recording's music bleed.

    Returns (offset_seconds, confidence). offset is negative when the beat
    starts after the recording does, matching the --offset convention.
    """
    sos = butter(4, [150, 1800], "band", fs=SR, output="sos")
    dec = 10  # work at 4.41 kHz; plenty for timing to ~2 ms
    a = sosfilt(sos, music.mean(axis=1))[::dec][: 90 * SR // dec]  # first 90 s is enough
    b = sosfilt(sos, beat.mean(axis=1))[::dec]
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    corr = np.abs(np.concatenate([corr[-(len(b) - 1):], corr[: len(a)]]))  # lags -(len(b)-1) .. len(a)-1
    lags = np.arange(len(corr)) - (len(b) - 1)  # beat sample 0 sits at recording sample `lag`
    # Songs repeat, so later repeats of a section can match almost as well as the
    # true start. Only consider the beat starting between 20 s before and 60 s
    # after the recording began, and on near-ties prefer the earliest start.
    rate = SR / dec
    window = (lags >= -20 * rate) & (lags <= 60 * rate)
    best = corr[window].max()
    candidates = np.nonzero(window & (corr >= 0.9 * best))[0]
    peak = candidates[np.argmin(np.abs(lags[candidates]))]
    confidence = float(corr[peak] / (np.median(corr) + 1e-12))
    return -lags[peak] / rate, confidence


def find_repeat(music: np.ndarray, beat: np.ndarray, expected: int) -> int | None:
    """Look for the beat being restarted near sample `expected` of the recording.

    Matches the first 20 s of the beat against the recording from just before
    the previous copy ends to 30 s after. Returns the start sample, or None if
    the beat can't be heard there.
    """
    sos = butter(4, [150, 1800], "band", fs=SR, output="sos")
    dec, head, min_overlap = 10, 20 * SR, 8 * SR
    lo = max(0, expected - SR // 2)
    hi = min(len(music), expected + 30 * SR + head)
    if hi - lo < min_overlap:
        return None
    a = sosfilt(sos, music[lo:hi].mean(axis=1))[::dec]
    b = sosfilt(sos, beat[:head].mean(axis=1))[::dec]
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.abs(np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)[: len(a)])
    corr[max(1, len(a) - min_overlap // dec):] = 0  # need 8 s of overlap to trust a match
    peak = int(np.argmax(corr))
    if corr[peak] / (np.median(corr) + 1e-12) < 20:
        return None
    return lo + peak * dec


def singing_end(vox: np.ndarray) -> int:
    """Sample where the last sung phrase ends (ignores trailing room noise)."""
    hop, win = SR // 100, SR // 33
    power = np.convolve(vox.mean(axis=1) ** 2, np.ones(win) / win, mode="same")[::hop]
    level = 10 * np.log10(power + 1e-12)
    noise, voice = np.percentile(level, 10), np.percentile(level, 95)
    active = np.nonzero(level > noise + 0.4 * (voice - noise))[0]
    return int((active[-1] + 1) * hop) if len(active) else len(vox)


def plan_beat(first_start: int, beat_len: int, vocal_len: int,
              music: np.ndarray | None, beat: np.ndarray) -> list[int]:
    """Where each copy of the beat starts (recording samples), repeating it
    until the singing is covered."""
    starts = [first_start]
    while starts[-1] + beat_len < vocal_len - SR:  # singing runs on past this copy
        expected = starts[-1] + beat_len
        found = find_repeat(music, beat, expected) if music is not None else None
        if found is not None and found > starts[-1] + SR:
            print(f"  Beat restarts at {found / SR:.2f}s in your recording; repeating it there")
            starts.append(found)
        else:
            print(f"  Recording is longer than the beat; repeating it back-to-back at {expected / SR:.2f}s")
            starts.append(expected)
    return starts


# Krumhansl-Schmuckler key profiles (relative weight of each scale degree).
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_KEY_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def detect_key(beat: np.ndarray) -> tuple[str, str]:
    """Estimate the beat's key from which pitch classes it uses most."""
    x = beat.mean(axis=1)[: 120 * SR]
    win, hop = 16384, 8192  # long window: enough resolution to tell bass notes apart
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * np.hanning(win)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    mag = (mag / (mag.max(axis=1, keepdims=True) + 1e-12)).mean(axis=0)  # loud bars don't dominate
    freqs = np.fft.rfftfreq(win, 1 / SR)
    keep = (freqs > 80) & (freqs < 2500)
    midi = 12 * np.log2(freqs[keep] / 440) + 69
    near = np.abs(midi - np.round(midi)) < 0.3  # ignore bins between semitones
    weight = mag[keep] * near * np.where(freqs[keep] < 250, 2.0, 1.0)  # bass outlines the key
    chroma = np.bincount(np.round(midi).astype(int) % 12, weights=weight, minlength=12)
    best = max(
        (np.corrcoef(chroma, np.roll(profile, root))[0, 1], _KEY_NAMES[root], scale)
        for root in range(12)
        for profile, scale in ((_MAJOR_PROFILE, "major"), (_MINOR_PROFILE, "minor"))
    )
    return best[1], best[2]


def rms_db(x: np.ndarray) -> float:
    active = x[np.abs(x).max(axis=1) > 1e-3]  # ignore silence when measuring level
    if len(active) == 0:
        return -120.0
    return 20 * np.log10(np.sqrt(np.mean(active ** 2)) + 1e-12)


def envelope(x: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Peak follower on a mono signal (vectorised enough to stay fast)."""
    mono = np.abs(x).max(axis=1)
    # Downsample the envelope calculation to 1 kHz, then interpolate back up.
    hop = SR // 1000
    blocks = mono[: len(mono) // hop * hop].reshape(-1, hop).max(axis=1)
    a, r = np.exp(-1 / (attack * 1000)), np.exp(-1 / (release * 1000))
    env = np.empty_like(blocks)
    level = 0.0
    for i, v in enumerate(blocks):
        coef = a if v > level else r
        level = coef * level + (1 - coef) * v
        env[i] = level
    return np.interp(np.arange(len(mono)), np.arange(len(env)) * hop, env)


def compress(x: np.ndarray, threshold_db=-18.0, ratio=3.5) -> np.ndarray:
    env_db = 20 * np.log10(envelope(x, 0.005, 0.12) + 1e-9)
    over = np.maximum(0, env_db - threshold_db)
    gain = 10 ** (-(over - over / ratio) / 20)
    return x * gain[:, None]


def reverb(x: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return x
    rng = np.random.default_rng(1)
    t = np.arange(int(SR * 1.8)) / SR
    ir = rng.standard_normal((len(t), 2)) * np.exp(-t * 3.2)[:, None]
    ir[:, 0] = sosfilt(butter(2, 6000, "low", fs=SR, output="sos"), ir[:, 0])
    ir[:, 1] = sosfilt(butter(2, 6000, "low", fs=SR, output="sos"), ir[:, 1])
    ir /= np.sqrt(np.sum(ir ** 2, axis=0))
    pre = int(SR * 0.025)  # short pre-delay keeps the voice up front
    wet = np.stack([fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], axis=1)
    wet = np.concatenate([np.zeros((pre, 2), np.float32), wet[:-pre]])
    return x + amount * wet


def gate(x: np.ndarray, floor_db: float = -26.0) -> np.ndarray:
    """Turn down breaths, room noise and leftover bleed between sung phrases."""
    hop, win = SR // 100, SR // 33  # 10 ms hop, 30 ms window
    power = np.convolve(x.mean(axis=1) ** 2, np.ones(win) / win, mode="same")[::hop]
    level = 10 * np.log10(power + 1e-12)
    # Threshold adapts to the recording: 40% of the way from its noise floor to its singing level.
    noise, voice = np.percentile(level, 10), np.percentile(level, 95)
    thresh = noise + 0.4 * (voice - noise)
    target = np.clip((level - (thresh - 6)) / 6, 0, 1)  # 6 dB soft knee
    # Open fast (no clipped word starts), close slowly (no chopped word endings).
    up, down = 1 - np.exp(-1 / 0.5), 1 - np.exp(-1 / 15)  # ~5 ms attack, ~150 ms release
    gain, g = np.empty_like(target), 0.0
    for i, t in enumerate(target):
        g += (up if t > g else down) * (t - g)
        gain[i] = g
    floor = 10 ** (floor_db / 20)
    gain = np.interp(np.arange(len(x)), np.arange(len(gain)) * hop, floor + (1 - floor) * gain)
    return (x * gain[:, None]).astype(np.float32)


def peaking_eq(x: np.ndarray, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """RBJ-cookbook peaking EQ band."""
    a_ = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / SR
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a_, -2 * np.cos(w0), 1 - alpha * a_]
    a = [1 + alpha / a_, -2 * np.cos(w0), 1 - alpha / a_]
    return lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0).astype(np.float32)


def vocal_eq(x: np.ndarray) -> np.ndarray:
    x = peaking_eq(x, 300, -3.5, 0.9)   # less mud / boxy room
    x = peaking_eq(x, 3500, 4.0, 0.8)   # presence: words cut through the beat
    return peaking_eq(x, 7500, -2.5, 2.0)  # tame harsh "s" sounds


def delayed(x: np.ndarray, seconds: float) -> np.ndarray:
    n = int(SR * seconds)
    return np.concatenate([np.zeros(n, x.dtype), x[: len(x) - n]])


def pan(mono: np.ndarray, position: float, gain_db: float) -> np.ndarray:
    """Place a mono layer in stereo: position -1 = left, 0 = centre, 1 = right."""
    angle = (position + 1) * np.pi / 4
    g = 10 ** (gain_db / 20)
    return np.stack([mono * np.cos(angle), mono * np.sin(angle)], axis=1) * g * np.sqrt(2)


def produce_vocal(vox: np.ndarray, args) -> np.ndarray:
    """Auto-tune the lead and, with --polish/--harmony, build the backing layers."""
    f0 = detect_pitch(vox.mean(axis=1).astype(np.float64))
    tune = lambda **kw: autotune(vox, args.key, args.scale, args.retune, f0=f0, **kw)[:, 0]  # noqa: E731
    out = pan(tune(), 0.0, 0.0)
    if args.polish:
        print("  Adding doubled vocals ...")
        out += pan(delayed(tune(cents=+9), 0.017), -0.9, -8.0)
        out += pan(delayed(tune(cents=-9), 0.026), +0.9, -8.0)
    if args.harmony:
        print("  Adding harmony a third above ...")
        harm = delayed(tune(degree=2), 0.011)
        out += pan(harm, -0.35, -13.0) + pan(delayed(harm, 0.009), +0.35, -13.0)
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vocals", type=Path, help="vocal recording, or a full song with --isolate")
    ap.add_argument("beat", type=Path, help="instrumental / beat file")
    ap.add_argument("-o", "--output", type=Path, help="output WAV (default: <vocals>_on_<beat>.wav)")
    ap.add_argument("--offset", type=float, default=0.0, help="seconds into the beat where the vocal starts")
    ap.add_argument("--vocal-db", type=float, default=0.0, help="vocal level relative to the beat, in dB")
    ap.add_argument("--reverb", type=float, default=0.25, help="reverb amount (0 = dry)")
    ap.add_argument("--isolate", action="store_true", help="strip music from the vocal file with Demucs first")
    ap.add_argument("--no-duck", action="store_true", help="don't dip the beat under the vocal")
    ap.add_argument("--align", action="store_true",
                    help="auto-detect --offset from beat audible in the recording (implies --isolate)")
    ap.add_argument("--autotune", action="store_true", help="pitch-correct the vocal to the key")
    ap.add_argument("--key", default="auto",
                    help="key for --autotune, e.g. D, F#, Bb (default: detect from the beat)")
    ap.add_argument("--scale", choices=SCALES, default=None,
                    help="scale for --autotune (default: detected with the key, else major)")
    ap.add_argument("--retune", type=float, default=0.0,
                    help="retune speed in ms: 0 = hard robotic snap, 40-80 = natural")
    ap.add_argument("--polish", action="store_true",
                    help="auto-tune + doubled vocals, noise gate, vocal EQ, slapback echo")
    ap.add_argument("--harmony", action="store_true", help="add a harmony a third above the lead")
    args = ap.parse_args()
    if args.polish or args.harmony:
        args.autotune = True
    if args.key != "auto" and args.key.upper() not in NOTE_NAMES:
        sys.exit(f"Unknown key '{args.key}'. Use auto, or e.g. C, D, F#, Bb.")

    for p in (args.vocals, args.beat):
        if not p.is_file():
            sys.exit(f"File not found: {p}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH; install it first.")

    # Name the output after the options used, so different versions don't overwrite each other.
    style = "polish" if args.polish else "autotune" if args.autotune else ""
    tags = "".join(f"_{t}" for t in (style, "harmony" if args.harmony else "") if t)
    out = args.output or args.vocals.with_name(f"{args.vocals.stem}_on_{args.beat.stem}{tags}.wav")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vocal_src, music_src = args.vocals, None
        if args.isolate or args.align:
            print("[1/4] Isolating the voice with Demucs (a few minutes on CPU) ...")
            vocal_src, music_src = isolate_vocals(args.vocals, tmp)
        print("[2/4] Loading audio ...")
        vox = load(vocal_src, tmp)
        beat = load(args.beat, tmp)
        music = load(music_src, tmp) if args.align else None
        if args.align:
            offset, confidence = find_offset(music, beat)
            if confidence < 20:
                print(f"  Couldn't hear the beat in the recording (confidence {confidence:.1f}); "
                      f"keeping --offset {args.offset:g}. Set it by hand if the timing is off.")
            else:
                print(f"  Beat found in the recording; using --offset {offset:.2f} "
                      f"(confidence {confidence:.0f})")
                args.offset = offset
            if confidence < 20:
                music = None  # can't hear the beat, so don't try to find repeats either

    if args.autotune and args.key == "auto":
        args.key, detected_scale = detect_key(beat)
        args.scale = args.scale or detected_scale
        print(f"  Detected key: {args.key} {args.scale} (override with --key / --scale)")
    args.scale = args.scale or "major"

    sung_until = singing_end(vox)  # measured on the raw vocal, before any effects
    if args.polish:
        vox = gate(vox)
    if args.autotune:
        print(f"[3/4] Auto-tuning to {args.key} {args.scale} (retune {args.retune:g} ms) ...")
        vox = produce_vocal(vox, args)
    print("[3/4] Processing vocal (clean-up, EQ, compression, reverb) ...")
    vox = sosfilt(butter(2, 90, "high", fs=SR, output="sos"), vox, axis=0).astype(np.float32)
    if args.polish:
        vox = vocal_eq(vox)
    vox = compress(vox)
    if args.polish:  # quick slapback echo, darker than the dry voice so it sits behind it
        slap = sosfilt(butter(2, 3000, "low", fs=SR, output="sos"), vox, axis=0)
        vox = vox + 0.22 * np.stack([delayed(slap[:, 0], 0.105), delayed(slap[:, 1], 0.115)], axis=1)
    vox = reverb(vox, args.reverb)

    # Lay out the timeline in recording time: the vocal runs 0..len(vox), and
    # each copy of the beat starts at starts[k] (repeated if the vocal outlasts it).
    starts = plan_beat(-int(round(args.offset * SR)), len(beat), sung_until, music, beat)
    t0 = starts[0]  # the mix begins where the first beat begins
    if len(starts) == 1:
        end = max(starts[0] + len(beat), len(vox))
    else:  # don't play a whole extra copy after the singing stops
        end = max(sung_until + 2 * SR, min(starts[-1] + len(beat), sung_until + 6 * SR))
    length = end - t0
    vox_line = np.zeros((length, 2), np.float32)
    v0, v1 = max(0, t0), min(len(vox), end)
    vox_line[v0 - t0: v1 - t0] = vox[v0:v1]
    beat_line = np.zeros((length, 2), np.float32)
    xfade = int(0.02 * SR)
    for k, s in enumerate(starts):
        stop = min(starts[k + 1] if k + 1 < len(starts) else s + len(beat), s + len(beat), end)
        piece = beat[: stop - s].copy()
        if k > 0:  # short crossfades so the restart doesn't click
            piece[:xfade] *= np.linspace(0, 1, min(xfade, len(piece)), dtype=np.float32)[:, None]
        if k + 1 < len(starts) and len(piece) > xfade:
            piece[-xfade:] *= np.linspace(1, 0, xfade, dtype=np.float32)[:, None]
        beat_line[s - t0: s - t0 + len(piece)] += piece

    # Level-match: vocal sits ~1 dB above the beat by default, then user offset.
    gain_db = rms_db(beat_line) - rms_db(vox_line) + 1.0 + args.vocal_db
    vox_line *= 10 ** (gain_db / 20)

    if not args.no_duck:  # dip the beat up to 3 dB while the vocal is active
        env = envelope(vox_line, 0.01, 0.3)
        duck = 1 - 0.29 * np.clip(env / (env.max() + 1e-9) * 2, 0, 1)
        beat_line *= duck[:, None]

    print("[4/4] Mixing and mastering ...")
    mix = beat_line + vox_line
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)  # gentle soft-clip limiter
    mix *= 0.95 / (np.abs(mix).max() + 1e-9)
    fade = min(len(mix), int(SR * 1.5))
    mix[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    sf.write(out, mix, SR, subtype="PCM_16")
    print(f"Done: {out}  ({length / SR:.1f}s)")


if __name__ == "__main__":
    main()

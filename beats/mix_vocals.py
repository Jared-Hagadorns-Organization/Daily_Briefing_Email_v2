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
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

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


def target_pitch(f0: np.ndarray, key: str, scale: str, retune_ms: float, hop: int) -> np.ndarray:
    """Snap each detected pitch to the nearest note in the key."""
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
    if retune_ms > 0:  # glide toward the target note instead of jumping
        alpha = 1 - np.exp(-hop / (SR * retune_ms / 1000))
        smooth, prev = np.empty_like(snapped), None
        for i, (s, m) in enumerate(zip(snapped, midi)):
            prev = m if prev is None else prev + alpha * (s - prev)
            smooth[i] = prev
        snapped = smooth
    out[voiced] = 440 * 2 ** ((snapped - 69) / 12)
    return out


def autotune(vox: np.ndarray, key: str, scale: str, retune_ms: float) -> np.ndarray:
    """Pitch-correct a vocal with TD-PSOLA (keeps the voice's tone, moves the notes)."""
    hop = 256
    x = vox.mean(axis=1).astype(np.float64)
    f0 = detect_pitch(x, hop)
    tgt = target_pitch(f0, key, scale, retune_ms, hop)
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
    ap.add_argument("--key", default="D", help="key for --autotune, e.g. D, F#, Bb (default D)")
    ap.add_argument("--scale", choices=SCALES, default="major", help="scale for --autotune")
    ap.add_argument("--retune", type=float, default=0.0,
                    help="retune speed in ms: 0 = hard robotic snap, 40-80 = natural")
    args = ap.parse_args()
    if args.key.upper() not in NOTE_NAMES:
        sys.exit(f"Unknown key '{args.key}'. Use e.g. C, D, F#, Bb.")

    for p in (args.vocals, args.beat):
        if not p.is_file():
            sys.exit(f"File not found: {p}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH; install it first.")

    out = args.output or args.vocals.with_name(f"{args.vocals.stem}_on_{args.beat.stem}.wav")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vocal_src, music_src = args.vocals, None
        if args.isolate or args.align:
            print("[1/4] Isolating the voice with Demucs (a few minutes on CPU) ...")
            vocal_src, music_src = isolate_vocals(args.vocals, tmp)
        print("[2/4] Loading audio ...")
        vox = load(vocal_src, tmp)
        beat = load(args.beat, tmp)
        if args.align:
            offset, confidence = find_offset(load(music_src, tmp), beat)
            if confidence < 20:
                print(f"  Couldn't hear the beat in the recording (confidence {confidence:.1f}); "
                      f"keeping --offset {args.offset:g}. Set it by hand if the timing is off.")
            else:
                print(f"  Beat found in the recording; using --offset {offset:.2f} "
                      f"(confidence {confidence:.0f})")
                args.offset = offset

    if args.autotune:
        print(f"[3/4] Auto-tuning to {args.key} {args.scale} (retune {args.retune:g} ms) ...")
        vox = autotune(vox, args.key, args.scale, args.retune)
    print("[3/4] Processing vocal (clean-up, compression, reverb) ...")
    vox = sosfilt(butter(2, 90, "high", fs=SR, output="sos"), vox, axis=0).astype(np.float32)
    vox = compress(vox)
    vox = reverb(vox, args.reverb)

    # Place the vocal on the timeline.
    shift = int(round(args.offset * SR))
    if shift < 0:
        vox = vox[-shift:]
        shift = 0
    length = max(len(beat), shift + len(vox))
    vox_line = np.zeros((length, 2), np.float32)
    vox_line[shift:shift + len(vox)] = vox
    beat_line = np.zeros((length, 2), np.float32)
    beat_line[: len(beat)] = beat

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

"""Overlay a vocal track on a beat/instrumental and save a finished WAV.

Accepts any audio or video format ffmpeg can read (wav, mp3, m4a, mp4, ...).
If the vocal file is a full song (e.g. a Suno download with its own music),
pass --isolate to strip its backing with Demucs first, keeping only the voice.

Usage:
    python mix_vocals.py vocals.wav beat_instrumental.wav
    python mix_vocals.py suno_song.mp3 beat_instrumental.wav --isolate
    python mix_vocals.py vocals.wav beat.wav --offset 4.2 --vocal-db 2 -o final.wav

Options worth knowing:
    --offset S     start the vocal S seconds into the beat (negative trims the
                   start of the vocal instead)
    --vocal-db D   make the vocal louder (+) or quieter (-) relative to the beat
    --reverb R     0 = dry, 0.25 = default room, 0.5 = big arena
    --no-duck      don't dip the beat slightly while the vocal is singing
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


def isolate_vocals(path: Path, tmp: Path) -> Path:
    """Run Demucs and return the path of the separated vocal stem."""
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-o", str(tmp / "sep"), str(path)],
        check=True,
    )
    return tmp / "sep" / "htdemucs" / path.stem / "vocals.wav"


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
    args = ap.parse_args()

    for p in (args.vocals, args.beat):
        if not p.is_file():
            sys.exit(f"File not found: {p}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH; install it first.")

    out = args.output or args.vocals.with_name(f"{args.vocals.stem}_on_{args.beat.stem}.wav")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vocal_src = args.vocals
        if args.isolate:
            print("[1/4] Isolating the voice with Demucs (a few minutes on CPU) ...")
            vocal_src = isolate_vocals(args.vocals, tmp)
        print("[2/4] Loading audio ...")
        vox = load(vocal_src, tmp)
        beat = load(args.beat, tmp)

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

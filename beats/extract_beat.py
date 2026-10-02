"""Extract the instrumental ("beat") from a local video or audio file.

Pipeline:
  1. ffmpeg pulls the audio track out of the video as a 44.1 kHz stereo WAV.
  2. Demucs (open-source AI stem separator) splits it into stems.
  3. The non-vocal stems are written out as a single WAV.

Only use this on media you own or are licensed to use. Separating a track
does not change who owns it, so get permission before publishing the result.

Setup (one time):
    pip install demucs soundfile
    # ffmpeg must be on PATH (brew install ffmpeg / apt install ffmpeg / winget install ffmpeg)

Usage:
    python extract_beat.py my_video.mp4
    python extract_beat.py my_video.mp4 -o beat.wav --mode drums
    python extract_beat.py song.mp3 --mode drums+bass --model htdemucs_ft
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Which Demucs stems make up each output mode.
MODES = {
    "instrumental": ["drums", "bass", "other"],  # everything except vocals
    "drums": ["drums"],
    "drums+bass": ["drums", "bass"],
}


def extract_audio(src: Path, wav: Path) -> None:
    """Pull the audio track out of a video (or re-encode an audio file) to WAV."""
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )


def separate(wav: Path, out_dir: Path, model: str, device: str | None) -> Path:
    """Run Demucs and return the folder holding the four stems."""
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(out_dir), str(wav)]
    if device:
        cmd[3:3] = ["-d", device]
    subprocess.run(cmd, check=True)
    return out_dir / model / wav.stem


def mix_stems(stem_dir: Path, stems: list[str], dest: Path) -> None:
    import numpy as np
    import soundfile as sf

    mix, sr = None, None
    for name in stems:
        data, sr = sf.read(stem_dir / f"{name}.wav", always_2d=True)
        mix = data if mix is None else mix + data
    peak = np.max(np.abs(mix))
    if peak > 0.99:  # avoid clipping after summing stems
        mix *= 0.99 / peak
    sf.write(dest, mix, sr, subtype="PCM_16")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="local video or audio file (mp4, mov, mkv, mp3, wav, ...)")
    ap.add_argument("-o", "--output", type=Path, help="output WAV (default: <input>_<mode>.wav)")
    ap.add_argument("--mode", choices=MODES, default="instrumental",
                    help="instrumental = all but vocals (default); drums = percussion only; drums+bass")
    ap.add_argument("--model", default="htdemucs",
                    help="Demucs model; htdemucs_ft is higher quality but ~4x slower")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], help="force a device (auto by default)")
    args = ap.parse_args()

    if not args.input.is_file():
        sys.exit(f"Input not found: {args.input}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH; install it first.")

    out = args.output or args.input.with_name(f"{args.input.stem}_{args.mode.replace('+', '_')}.wav")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav = tmp / "source.wav"
        print(f"[1/3] Extracting audio from {args.input.name} ...")
        extract_audio(args.input, wav)
        print(f"[2/3] Separating stems with Demucs ({args.model}); this can take a few minutes on CPU ...")
        stem_dir = separate(wav, tmp / "stems", args.model, args.device)
        print(f"[3/3] Mixing {', '.join(MODES[args.mode])} -> {out}")
        mix_stems(stem_dir, MODES[args.mode], out)

    print(f"Done: {out}")


if __name__ == "__main__":
    main()

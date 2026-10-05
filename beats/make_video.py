"""Loop a short video clip for the length of a song and save it as an MP4.

The clip's own sound is dropped; the song plays underneath. The result plays
on phones, in group chats and on social apps.

Usage:
    python make_video.py dance.mp4 song.wav
    python make_video.py dance.mp4 song.wav --boomerang --title "Week 4 Recap"
    python make_video.py dance.mp4 song.wav --format vertical -o week4.mp4

Options:
    --boomerang      play the clip forward then backward each loop, so the
                     jump back to the start isn't noticeable
    --format F       original (default) keeps your clip's shape; vertical
                     (1080x1920, TikTok/Reels/Stories), square (1080x1080), or
                     landscape (1920x1080). The clip is fitted with a blurred
                     copy of itself filling any empty space.
    --title TEXT     caption across the top for the whole video
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SIZES = {"vertical": (1080, 1920), "square": (1080, 1080), "landscape": (1920, 1080)}
FPS = 30


def ffmpeg(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True, cwd=cwd)


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def dimensions(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def fit_title(text: str, width: int, height: int) -> tuple[str, int]:
    """Wrap the caption onto up to two lines and pick a font size that fits."""
    char_w = 0.6  # average bold-font character width, as a fraction of font size
    size = height // 18
    if len(text) * char_w * size > 0.92 * width:  # too wide for one line: break near the middle
        words, best = text.split(), None
        for i in range(1, len(words)):
            lines = [" ".join(words[:i]), " ".join(words[i:])]
            if best is None or max(map(len, lines)) < max(map(len, best)):
                best = lines
        if best:
            text = "\n".join(best)
    longest = max(len(line) for line in text.split("\n"))
    size = min(size, int(0.92 * width / (char_w * max(longest, 1))))
    return text, size


def hdr_to_sdr(clip: Path) -> str:
    """Filter prefix converting iPhone/Android HDR video to normal colours.

    Phones record HDR (HLG or PQ) by default; converted naively it looks grey
    and washed out on everything else. Returns "" for ordinary video.
    """
    transfer = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_transfer",
         "-of", "csv=p=0", str(clip)], capture_output=True, text=True,
    ).stdout.strip().split(",")[0]  # rotated phone clips append side-data fields
    if transfer not in ("arib-std-b67", "smpte2084"):
        return ""
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    if " zscale " not in filters or " tonemap " not in filters:
        print("  Note: this is an HDR clip, but this ffmpeg can't convert HDR; colours may look washed out.")
        return ""
    print("  HDR clip detected; converting to standard colours ...")
    return ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
            "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,")


def font_file() -> Path | None:
    """A bold system font for the title."""
    for f in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(f).is_file():
            return Path(f)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path, help="your short video (mp4, mov, ...)")
    ap.add_argument("song", type=Path, help="the finished song (wav, mp3, ...)")
    ap.add_argument("-o", "--output", type=Path, help="output MP4 (default: <song>_video.mp4)")
    ap.add_argument("--boomerang", action="store_true", help="forward-then-backward loop")
    ap.add_argument("--format", choices=["original", *SIZES], default="original", help="video shape")
    ap.add_argument("--title", help="caption shown across the top")
    args = ap.parse_args()

    for p in (args.clip, args.song):
        if not p.is_file():
            sys.exit(f"File not found: {p}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg not found on PATH; install it first.")

    out = (args.output or args.song.with_name(f"{args.song.stem}_video.mp4")).resolve()
    args.clip, args.song = args.clip.resolve(), args.song.resolve()
    song_len = duration(args.song)

    # Step 1: one clean loop of the clip at a steady frame rate (phones record
    # variable frame rates, which stutter when looped), resized if asked.
    hdr = hdr_to_sdr(args.clip)
    if args.format == "original":
        # Cap the long side at 1920 and keep even dimensions for H.264.
        shape = ("scale='if(gt(iw,ih),min(1920,iw),-2)':'if(gt(iw,ih),-2,min(1920,ih))',"
                 "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1")
        fit = f"[0:v]{hdr}fps={FPS},{shape}[v]"
    else:
        w, h = SIZES[args.format]
        fit = (f"[0:v]{hdr}fps={FPS},split[a][b];"
               f"[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},gblur=sigma=30,"
               f"eq=brightness=-0.08[bg];"
               f"[b]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
               f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
    if args.boomerang:
        fit = fit.replace("[v]", "[f]") + ";[f]split[x][y];[y]reverse,trim=start_frame=1[r];[x][r]concat=n=2:v=1[v]"

    with tempfile.TemporaryDirectory() as tmp:
        unit = Path(tmp) / "loop_unit.mp4"
        print("[1/2] Preparing your clip ...")
        ffmpeg("-i", str(args.clip), "-filter_complex", fit, "-map", "[v]", "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(unit))
        loops = int(song_len // duration(unit)) + 1

        # Step 2: loop it under the song, fade in/out, optional title.
        print(f"[2/2] Looping it {loops}x across the {song_len:.0f}s song and encoding ...")
        fade_out = max(0.0, song_len - 1.5)
        vf = f"fade=in:st=0:d=0.6,fade=out:st={fade_out:.2f}:d=1.5"
        if args.title:
            # Caption and font go in files next to the temp clip and are referenced by
            # bare name (ffmpeg runs in that folder), so no character needs escaping.
            text, size = fit_title(args.title, *dimensions(unit))
            (Path(tmp) / "title.txt").write_text(text, encoding="utf-8")
            font = font_file()
            if font:
                shutil.copy(font, Path(tmp) / "title.ttf")
            vf += (",drawtext=textfile=title.txt:expansion=none:" + ("fontfile=title.ttf:" if font else "") +
                   f"fontcolor=white:fontsize={size}:line_spacing={size // 5}:text_align=center:"
                   f"borderw={max(2, size // 14)}:bordercolor=black@0.85:x=(w-text_w)/2:y=h*0.06")
        ffmpeg("-stream_loop", "-1", "-i", str(unit), "-i", str(args.song),
               "-map", "0:v", "-map", "1:a", "-t", f"{song_len:.3f}",
               "-vf", vf, "-af", f"afade=out:st={fade_out:.2f}:d=1.5",
               "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out), cwd=Path(tmp))

    print(f"Done: {out}  ({song_len:.1f}s)")


if __name__ == "__main__":
    main()

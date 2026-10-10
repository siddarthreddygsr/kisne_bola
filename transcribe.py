import argparse
import sys
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()
MODEL_ID = "bodhan-ai/indic-transcribe-core"
SHORT_CLIP_SECONDS = 40
OUT_DIR = Path(__file__).resolve().parent / "out"


def load_model():
    try:
        resolved = Path(snapshot_download(MODEL_ID))
    except Exception as exc:
        raise SystemExit(f"Error loading model: {exc}") from exc

    sys.path.insert(0, str(resolved))
    from indic_transcribe import IndicTranscribe
    from long_form import transcribe_long

    print(f"Loading {MODEL_ID}", file=sys.stderr)
    return IndicTranscribe.from_pretrained(str(resolved)), transcribe_long


def transcribe_file(asr, transcribe_long, path: Path) -> None:
    info = sf.info(str(path))
    if info.frames / info.samplerate <= SHORT_CLIP_SECONDS:
        text, lid = asr.transcribe(str(path), return_lid=True)
        print(f"[{path.name}] lang={lid.get('lang', 'unknown')}", file=sys.stderr)
    else:
        text = transcribe_long(asr, str(path))
        ranked = asr.identify(str(path))
        detected = ranked[0][0] if ranked else "unknown"
        print(f"[{path.name}] lang={detected}", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUT_DIR / f"{path.stem}.txt"
    destination.write_text(text.strip() + "\n", encoding="utf-8")
    print(destination, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=Path, required=True, help="wav, flac, or mp3 file")
    audio = parser.parse_args().i
    if not audio.is_file():
        raise SystemExit(f"Audio file not found: {audio}")

    asr, transcribe_long = load_model()
    transcribe_file(asr, transcribe_long, audio)


if __name__ == "__main__":
    main()

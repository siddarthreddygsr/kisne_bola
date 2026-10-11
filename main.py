import argparse
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

load_dotenv()
DIAR_MODEL_ID = "nvidia/Nemotron-3-Diarization"
ASR_MODELS = {
    "ai4bharat": "bodhan-ai/indic-transcribe-core",
    "sravaani": "ARTPARK-IISc/SraVaani-1.0",
}
SHORT_CLIP_SECONDS = 40
OUT_DIR = Path(__file__).resolve().parent / "out"


def load_models(stt: str):
    print(f"Loading {DIAR_MODEL_ID}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(DIAR_MODEL_ID)
    diar_model = AutoModelForAudioFrameClassification.from_pretrained(DIAR_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diar_model = diar_model.to(device)
    diar_model.eval()
    print(f"device={device}", file=sys.stderr)

    model_id = ASR_MODELS[stt]
    print(f"Loading {model_id}", file=sys.stderr)
    if stt == "sravaani":
        kwargs = {"trust_remote_code": True}
        if device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        try:
            asr = AutoModel.from_pretrained(model_id, **kwargs).to(device).eval()
        except Exception as exc:
            raise SystemExit(
                f"Could not load {model_id}. Accept the model terms at "
                f"https://huggingface.co/{model_id} and set HF_TOKEN. {exc}"
            ) from exc
        return processor, diar_model, asr, None

    try:
        resolved = Path(snapshot_download(model_id))
    except Exception as exc:
        raise SystemExit(f"Error loading model: {exc}") from exc

    sys.path.insert(0, str(resolved))
    from indic_transcribe import IndicTranscribe
    from long_form import transcribe_long

    asr = IndicTranscribe.from_pretrained(str(resolved))
    return processor, diar_model, asr, transcribe_long


def diarize(processor, model, path: Path) -> list[dict]:
    sampling_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(str(path), sampling_rate=sampling_rate)
    inputs = processor(audio, sampling_rate=sampling_rate).to(
        model.device, dtype=model.dtype
    )
    with torch.inference_mode():
        logits = model(**inputs).logits
    return processor.extract_speaker_dict(logits, inputs.attention_mask)[0]


def transcribe_span(
    asr, transcribe_long, source: Path, start: float, end: float, lang, clip: Path
) -> str:
    info = sf.info(str(source))
    start_frame = max(0, int(start * info.samplerate))
    end_frame = min(info.frames, int(end * info.samplerate))
    if end_frame <= start_frame:
        return ""

    audio, sample_rate = sf.read(
        str(source), start=start_frame, stop=end_frame, always_2d=True
    )
    sf.write(clip, audio.mean(axis=1), sample_rate)
    if transcribe_long is None:
        duration = (end_frame - start_frame) / sample_rate
        if duration <= 30:
            hypotheses = asr.transcribe(str(clip), return_hypotheses=True)
            text = hypotheses[0].text if hypotheses else ""
            return str(text).strip()
        parts = []
        piece = clip.with_name("piece.wav")
        offset = 0.0
        while offset < duration - 0.05:
            piece_end = min(duration, offset + 30)
            chunk, rate = sf.read(
                str(clip),
                start=int(offset * sample_rate),
                stop=int(piece_end * sample_rate),
                always_2d=True,
            )
            sf.write(piece, chunk.mean(axis=1), rate)
            hypotheses = asr.transcribe(str(piece), return_hypotheses=True)
            text = hypotheses[0].text if hypotheses else ""
            parts.append(str(text).strip())
            offset = piece_end
        return " ".join(part for part in parts if part)
    if (end_frame - start_frame) / sample_rate <= SHORT_CLIP_SECONDS:
        text = asr(str(clip), lang=lang) if lang else asr(str(clip))
    else:
        text = transcribe_long(asr, str(clip), lang=lang)
    return str(text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=Path, required=True, help="wav, flac, or mp3 file")
    parser.add_argument(
        "--stt",
        choices=sorted(ASR_MODELS),
        default="ai4bharat",
        help="ai4bharat or sravaani",
    )
    args = parser.parse_args()
    audio = args.i
    if not audio.is_file():
        raise SystemExit(f"Audio file not found: {audio}")

    processor, diar_model, asr, transcribe_long = load_models(args.stt)
    segments = diarize(processor, diar_model, audio)
    print(segments, file=sys.stderr)
    del processor, diar_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if transcribe_long is None:
        lang = None
        print(f"[{audio.name}] stt=sravaani", file=sys.stderr)
    else:
        ranked = asr.identify(str(audio))
        lang = ranked[0][0] if ranked else None
        print(f"[{audio.name}] lang={lang or 'unknown'}", file=sys.stderr)

    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.wav"
        for segment in segments:
            text = transcribe_span(
                asr,
                transcribe_long,
                audio,
                segment["Start"],
                segment["End"],
                lang,
                clip,
            )
            speaker = f"speaker_{segment['Speaker']}"
            lines.append(
                f"{speaker} [{segment['Start']:.2f}-{segment['End']:.2f}] {text}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = (
        f"{audio.stem}.sravaani.txt" if args.stt == "sravaani" else f"{audio.stem}.txt"
    )
    destination = OUT_DIR / name
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(destination, file=sys.stderr)


if __name__ == "__main__":
    main()

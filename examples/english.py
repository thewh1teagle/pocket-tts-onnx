"""Generate audio.wav from text with the English model.

The English model is fetched from the Hub on first run and cached (or see
docs/EXPORT.md to build one and pass its path to `PocketTTS` instead). Then:

    uv run python examples/english.py
"""

import soundfile as sf

from pocket_tts_onnx import PocketTTS

MODEL = "english"
TEXT = (
    "Hello world. I am Kyutai's Pocket TTS, now running on onnxruntime. "
    "I stream audio frame by frame, and there is no torch anywhere in sight."
)


def main() -> None:
    tts = PocketTTS.from_pretrained(MODEL)
    print("voices:", ", ".join(tts.voices()))

    samples, sample_rate = tts.create(TEXT, voice="alba")
    sf.write("audio.wav", samples, sample_rate)
    print(f"wrote audio.wav, {len(samples) / sample_rate:.2f}s at {sample_rate} Hz")

    # Cloning is a separate step: encode a prompt once, then synthesise with it.
    #
    #   cond = tts.clone_voice("my_voice.wav")
    #   samples, sample_rate = tts.create(TEXT, voice=cond)


if __name__ == "__main__":
    main()

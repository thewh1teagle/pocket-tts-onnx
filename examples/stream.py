"""Stream straight to the speakers, so you can feel the latency.

The English model is fetched from the Hub on first run and cached. Playback
needs sounddevice:

    uv add sounddevice

Then:

    uv run python examples/stream.py
"""

import time

import numpy as np
import sounddevice as sd

from pocket_tts_onnx import PocketTTS

MODEL = "english"
VOICE = "alba"
# `write` returns once a frame is queued, not once it is played, so closing the
# stream right after the last one leaves PortAudio short of data and it pops.
# A little silence gives it something to finish on.
TAIL_SECONDS = 0.15
DEFAULT_TEXT = (
    "Hello! I am streaming this to you frame by frame, "
    "eighty milliseconds at a time, as I generate it."
)


def speak(tts: PocketTTS, text: str) -> None:
    """Play frames as they arrive, and report when the first one landed."""
    started = time.perf_counter()
    first = None
    with sd.OutputStream(samplerate=tts.sample_rate, channels=1, dtype="float32") as speaker:
        for frame in tts.stream(text, voice=VOICE):
            if first is None:
                first = time.perf_counter() - started
                print(f"  first audio after {first * 1000:.0f} ms, still generating the rest")
            speaker.write(frame)
        speaker.write(np.zeros(int(TAIL_SECONDS * tts.sample_rate), dtype="float32"))
    print(f"  done in {time.perf_counter() - started:.2f}s")


def main() -> None:
    print(f"Loading {MODEL}...")
    tts = PocketTTS.from_pretrained(MODEL)

    # The first call pays for onnxruntime warming up and for prompting the
    # voice. Doing it now, silently, is what a server would do at startup.
    for _ in tts.stream("Warming up.", voice=VOICE):
        pass
    print("Warm.\n")

    print("Type something and press enter to hear it streamed.")
    print("Press enter on its own for the default line, or type 'q' to quit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {"q", "quit", "exit"}:
            return
        speak(tts, text or DEFAULT_TEXT)


if __name__ == "__main__":
    main()

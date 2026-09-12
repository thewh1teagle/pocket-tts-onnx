"""The same sentence a few ways: temperature, decode steps, and the seed.

The English model is fetched from the Hub on first run and cached (or see
docs/EXPORT.md to build one and pass its path to `PocketTTS` instead). Then:

    uv run python examples/tuning.py

Writes one wav per setting, so they can be played side by side.
"""

import time

import soundfile as sf

from pocket_tts_onnx import PocketTTS

MODEL = "english"
TEXT = "Well, that was unexpected. Are you sure you want to go through with it?"


def main() -> None:
    tts = PocketTTS.from_pretrained(MODEL)

    # Temperature scales the noise the flow sampler starts from. It is the one
    # knob that moves prosody: low is flat and safe, high is lively and, past
    # about 0.6, starts to stumble. The model's own default is 0.3.
    for temperature in (0.15, 0.3, 0.5):
        samples, rate = tts.create(TEXT, voice="alba", temperature=temperature, seed=1)
        sf.write(f"tuning-temperature-{temperature}.wav", samples, rate)
        print(f"temperature {temperature}: {len(samples) / rate:.2f}s")

    # Decode steps are how many times the flow head refines each frame, 1 to
    # 4. More is cleaner and slower; 2 is the usual trade, and the difference
    # above that is in the last bits.
    for steps in (1, 2, 4):
        started = time.perf_counter()
        samples, rate = tts.create(TEXT, voice="alba", decode_steps=steps, seed=1)
        elapsed = time.perf_counter() - started
        sf.write(f"tuning-steps-{steps}.wav", samples, rate)
        print(f"decode steps {steps}: {len(samples) / rate:.2f}s of audio in {elapsed:.2f}s")

    # The seed drives the sampler's noise. The same text, voice and seed give
    # the same take; a different seed is a different reading of it. When one
    # take is not quite right, this is the cheapest thing to change.
    for seed in (1, 2, 3):
        samples, rate = tts.create(TEXT, voice="alba", seed=seed)
        sf.write(f"tuning-seed-{seed}.wav", samples, rate)
        print(f"seed {seed}: {len(samples) / rate:.2f}s")


if __name__ == "__main__":
    main()

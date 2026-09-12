"""Faster or slower, after the fact, with the pitch left alone.

The model has no speaking-rate control, so speed is a post-process on the
audio. This is WSOLA — overlap-add with a small search for the best-aligned
window — in a few lines of numpy, which is plenty for 0.7x to 1.4x on speech.
The English model is fetched from the Hub on first run and cached. Then:

    uv run python examples/speed.py
"""

import numpy as np
import soundfile as sf

from pocket_tts_onnx import PocketTTS

MODEL = "english"
TEXT = "This is the same take three times: a little slower, as it was, and a little faster."


def stretch(samples: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
    """Play `samples` `rate` times faster (2.0 halves the length), pitch preserved.

    Windows of about 40 ms are taken from the input at a stride of `rate` times
    the output stride and overlap-added; each is nudged by up to ±10 ms to the
    offset that best matches the tail of what has been written, so the joins
    fall on the waveform's own period rather than across it.
    """
    if abs(rate - 1.0) < 1e-3:
        return samples
    window = int(0.040 * sample_rate)
    hop = window // 2
    search = int(0.010 * sample_rate)
    taper = np.hanning(window).astype(np.float32)

    length = int(len(samples) / rate)
    out = np.zeros(length + window, dtype=np.float32)
    norm = np.zeros(length + window, dtype=np.float32)

    position = 0.0
    for start in range(0, length, hop):
        centre = int(position)
        lo = max(0, centre - search)
        hi = min(len(samples) - window, centre + search)
        if hi <= lo:
            break
        # The best of the candidate windows is the one that continues what
        # the previous window left in the overlap.
        target = out[start : start + hop]
        best, best_score = lo, -np.inf
        for offset in range(lo, hi + 1, max(1, search // 16)):
            score = float(np.dot(samples[offset : offset + hop], target))
            if score > best_score:
                best, best_score = offset, score
        out[start : start + window] += samples[best : best + window] * taper
        norm[start : start + window] += taper
        position += hop * rate

    norm[norm < 1e-3] = 1.0
    return (out / norm)[:length]


def main() -> None:
    tts = PocketTTS.from_pretrained(MODEL)
    samples, sample_rate = tts.create(TEXT, voice="alba", seed=1)
    samples = np.asarray(samples, dtype=np.float32)

    for rate in (0.8, 1.0, 1.25):
        stretched = stretch(samples, rate, sample_rate)
        sf.write(f"speed-{rate}.wav", stretched, sample_rate)
        print(f"{rate}x: {len(stretched) / sample_rate:.2f}s")


if __name__ == "__main__":
    main()

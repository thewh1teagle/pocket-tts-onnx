"""Streaming pocket-tts inference on onnxruntime.

The exported graph is one streaming step: text embeddings, voice conditioning or
the previously generated latent go in, one latent and one 80 ms audio frame come
out, along with the keys/values and convolution state that step produced. This
module drives that graph, keeping the caches in preallocated numpy buffers so a
step only ever hands the runtime a contiguous window of the past.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pocket_tts_onnx import hub
from pocket_tts_onnx.audio import pad_to_frames, read_audio, resample
from pocket_tts_onnx.text import (
    MixedTokenizer,
    prepare_phoneme_prompt,
    prepare_text_prompt,
    split_into_best_sentences,
    split_phoneme_chunks,
)

_TEXT, _LATENT, _COND = (
    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 1.0], dtype=np.float32),
)


@dataclass
class _Config:
    sample_rate: int
    frame_size: int
    frame_rate: float
    latent_dim: int
    model_dim: int
    flow_layers: int
    flow_heads: int
    flow_head_dim: int
    mimi_layers: int
    mimi_heads: int
    mimi_head_dim: int
    mimi_kv_len: int
    mimi_steps_per_latent: int
    conv_state_size: int
    sampler_decode_steps: int
    temperature: float
    noise_clamp: float | None
    eos_threshold: float
    max_tokens_per_chunk: int
    pad_with_spaces_for_short_inputs: bool
    remove_semicolons: bool
    frames_after_eos: int | None
    tokens_per_second_estimate: float
    gen_seconds_padding: float
    language: str | None
    voices: dict
    voice_languages: dict | None = None
    lora: dict | None = None
    max_decode_steps: int = 1
    quantized: bool = False


class _Cache:
    """A grow-once buffer whose tail is handed to the graph as the past."""

    def __init__(self, shape: tuple[int, ...], capacity: int):
        self.buffer = np.zeros((capacity, *shape), dtype=np.float32)
        self.shape = shape
        self.length = 0

    def reserve(self, extra: int) -> None:
        needed = self.length + extra
        if needed > len(self.buffer):
            grown = np.zeros((max(needed, 2 * len(self.buffer)), *self.shape), dtype=np.float32)
            grown[: self.length] = self.buffer[: self.length]
            self.buffer = grown

    def append(self, values: np.ndarray) -> None:
        self.reserve(len(values))
        self.buffer[self.length : self.length + len(values)] = values
        self.length += len(values)

    def window(self, size: int | None = None) -> np.ndarray:
        """The last `size` entries, zero-padded at the front while still short."""
        if size is None:
            return self.buffer[: self.length]
        start = self.length - size
        if start >= 0:
            return self.buffer[start : self.length]
        padded = np.zeros((size, *self.shape), dtype=np.float32)
        if self.length:
            padded[size - self.length :] = self.buffer[: self.length]
        return padded

    def truncate(self, length: int) -> None:
        self.length = length


class PocketTTS:
    """Text to speech from a single exported pocket-tts ONNX file.

    ```python
    tts = PocketTTS("pocket-tts-english.onnx")     # a file you have
    tts = PocketTTS.from_pretrained("english")     # fetched from the Hub, then cached
    samples, sample_rate = tts.create("Hello world.", voice="alba")
    ```
    """

    def __init__(
        self,
        path: str | Path,
        providers: list[str] | None = None,
        num_threads: int = 2,
    ):
        import sentencepiece

        self.path = Path(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.path), options, providers=providers or ["CPUExecutionProvider"]
        )
        self._output_names = [output.name for output in self.session.get_outputs()]
        self._input_names = {graph_input.name for graph_input in self.session.get_inputs()}

        metadata = self.session.get_modelmeta().custom_metadata_map
        self.config = _Config(**json.loads(metadata["pocket_tts_config"]))
        self._voice_blobs = {
            key.removeprefix("pocket_tts_voice/"): value
            for key, value in metadata.items()
            if key.startswith("pocket_tts_voice/")
        }
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=base64.b64decode(metadata["pocket_tts_tokenizer"])
        )

        self.phonemes_tokenizer: MixedTokenizer | None = None
        if self.config.lora is not None:
            self.phonemes_tokenizer = MixedTokenizer(
                self.tokenizer,
                self.config.lora.get("atomic_chars") or self.config.lora["ipa_chars"],
                self.config.lora["vocab_base"],
            )
        self._encoder_blob = metadata.get("pocket_tts_encoder")
        self._encoder: ort.InferenceSession | None = None
        self._session_options = options
        self._providers = providers or ["CPUExecutionProvider"]
        self._voice_cache: dict[tuple[str, float], tuple[np.ndarray, int]] = {}

    @classmethod
    def from_pretrained(cls, name: str = "english", repo: str = hub.REPO, **kwargs) -> "PocketTTS":
        """A model by name from the Hub, downloaded on first use and cached.

        `name` is `english`, `english-ipa` (with the Hebrew adapter bundled in),
        or the exact filename in `repo`. Keyword arguments go to the constructor.
        """
        return cls(hub.download(name, repo), **kwargs)

    # ------------------------------------------------------------------ voices

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def voices(self) -> list[str]:
        """Names of the voices carried inside the ONNX file."""
        return sorted(self._voice_blobs)

    def voice_conditioning(self, voice: str | np.ndarray) -> np.ndarray:
        """Voice conditioning as [1, T, model_dim].

        Takes a built-in voice name or an array from `clone_voice`. Cloning is a
        separate call on purpose: synthesis never touches the encoder.
        """
        if isinstance(voice, np.ndarray):
            array = voice
        else:
            if voice not in self._voice_blobs:
                extra = (
                    " (to use an audio file, clone it first with clone_voice)"
                    if Path(voice).suffix
                    else ""
                )
                raise KeyError(f"unknown voice {voice!r}; available: {self.voices()}{extra}")
            shape = self.config.voices[voice]
            array = np.frombuffer(
                base64.b64decode(self._voice_blobs[voice]), dtype=np.float16
            ).reshape(shape)
        array = np.ascontiguousarray(array, dtype=np.float32)
        return array[None] if array.ndim == 2 else array

    def clone_voice(
        self,
        audio: str | Path | np.ndarray,
        sample_rate: int | None = None,
        max_seconds: float = 20.0,
    ) -> np.ndarray:
        """Encode a voice prompt into conditioning usable as `voice=`.

        `audio` is a path to an audio file, or mono float32 samples together
        with their `sample_rate`. The returned [1, T, model_dim] array is what
        `create` / `stream` take as `voice=`; save it with `np.save` and reuse
        it, so the encoder runs once per voice and never during synthesis.
        """
        if self._encoder_blob is None:
            raise RuntimeError(
                f"{self.path.name} carries no voice encoder; re-export with a "
                "current pocket-tts-onnx to clone voices"
            )
        if isinstance(audio, (str, Path)):
            samples, rate = read_audio(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate is required when passing samples")
            samples, rate = np.asarray(audio, dtype=np.float32), sample_rate
        if samples.ndim > 1:
            samples = samples.mean(axis=tuple(range(samples.ndim - 1)))

        samples = resample(samples, rate, self.config.sample_rate)
        limit = int(max_seconds * self.config.sample_rate)
        if len(samples) > limit:
            samples = samples[:limit]
        samples = pad_to_frames(samples, self.config.frame_size)

        if self._encoder is None:
            self._encoder = ort.InferenceSession(
                base64.b64decode(self._encoder_blob), self._session_options, self._providers
            )
        (cond,) = self._encoder.run(None, {"audio": samples[None, None]})
        return cond

    # -------------------------------------------------------------- inference

    def create(
        self,
        text: str,
        voice: str | np.ndarray | None = "alba",
        phonemes: bool = False,
        temperature: float | None = None,
        decode_steps: int | None = None,
        seed: int | None = None,
        lora: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Generate speech for `text` and return `(samples, sample_rate)`.

        `samples` is a mono float32 array in [-1, 1]. Set `phonemes=True` to feed
        stressed IPA instead of ordinary spelling; that also switches the model's
        adapter on, if the file carries one.
        """
        chunks = list(
            self.stream(
                text,
                voice=voice,
                phonemes=phonemes,
                temperature=temperature,
                decode_steps=decode_steps,
                seed=seed,
                lora=lora,
            )
        )
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        return np.concatenate(chunks), self.sample_rate

    def stream(
        self,
        text: str,
        voice: str | np.ndarray | None = "alba",
        phonemes: bool = False,
        temperature: float | None = None,
        decode_steps: int | None = None,
        seed: int | None = None,
        lora: float | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield 80 ms mono float32 frames as they are decoded.

        `phonemes=True` reads the text as stressed IPA. `lora` overrides the
        adapter gate, which otherwise follows `phonemes`.
        """
        config = self.config
        tokenizer, defaults = self._frontend(phonemes)
        temperature = (
            defaults.get("temperature", config.temperature) if temperature is None else temperature
        )
        gate = float(phonemes) if lora is None else float(lora)
        steps = config.sampler_decode_steps if decode_steps is None else decode_steps
        if not 1 <= steps <= config.max_decode_steps:
            raise ValueError(
                f"decode_steps must be between 1 and {config.max_decode_steps} "
                f"for this file, got {steps}"
            )
        steps = float(steps)
        max_tokens = defaults.get("max_tokens_per_chunk", config.max_tokens_per_chunk)
        rng = np.random.default_rng(seed)

        flow_kv, voice_length = self._prefilled_voice(voice, gate)
        if phonemes:
            chunks = split_phoneme_chunks(tokenizer, text, max_tokens)
        else:
            chunks = split_into_best_sentences(
                tokenizer,
                text,
                max_tokens,
                config.pad_with_spaces_for_short_inputs,
                config.remove_semicolons,
            )

        for chunk in chunks:
            if phonemes:
                prompt, guess = prepare_phoneme_prompt(chunk), 0
            else:
                prompt, guess = prepare_text_prompt(
                    chunk, config.pad_with_spaces_for_short_inputs, config.remove_semicolons
                )
            frames_after_eos = defaults.get("frames_after_eos", config.frames_after_eos)
            if frames_after_eos is None:
                frames_after_eos = guess + 2
            tokens = np.asarray([tokenizer.encode(prompt, out_type=int)], dtype=np.int64)
            max_frames = self._max_frames(tokens.shape[1])

            flow_kv.truncate(voice_length)
            flow_kv.reserve(tokens.shape[1] + max_frames)
            mimi_kv = _Cache(
                (config.mimi_layers, 2, 1, config.mimi_heads, config.mimi_head_dim),
                config.mimi_kv_len + max_frames * config.mimi_steps_per_latent,
            )
            mimi_conv = np.zeros(config.conv_state_size, dtype=np.float32)
            mimi_offset = np.zeros((), dtype=np.int64)

            outputs = self._run(
                tokens=tokens,
                gates=_TEXT,
                seq=tokens.shape[1],
                noise=np.zeros((1, config.latent_dim), dtype=np.float32),
                flow_kv=flow_kv,
                mimi_kv=mimi_kv,
                mimi_offset=mimi_offset,
                mimi_conv=mimi_conv,
                decode_steps=steps,
                lora=gate,
            )
            flow_kv.append(outputs["flow_kv_new"])

            latent = np.zeros((1, 1, config.latent_dim), dtype=np.float32)
            is_bos = np.ones((1, 1, 1), dtype=np.float32)
            eos_frame = None
            for frame in range(max_frames):
                noise = self._noise(rng, temperature)
                outputs = self._run(
                    latent=latent,
                    is_bos=is_bos,
                    gates=_LATENT,
                    seq=1,
                    noise=noise,
                    flow_kv=flow_kv,
                    mimi_kv=mimi_kv,
                    mimi_offset=mimi_offset,
                    mimi_conv=mimi_conv,
                    decode_steps=steps,
                    lora=gate,
                )
                flow_kv.append(outputs["flow_kv_new"])
                mimi_kv.append(outputs["mimi_kv_new"])
                mimi_conv = outputs["mimi_conv_out"]
                mimi_offset = outputs["mimi_offset_out"]
                latent = outputs["next_latent"]
                is_bos = np.zeros((1, 1, 1), dtype=np.float32)

                if eos_frame is None and outputs["eos_logit"].item() > config.eos_threshold:
                    eos_frame = frame
                if eos_frame is not None and frame >= eos_frame + frames_after_eos:
                    break
                yield outputs["audio"][0, 0]

    # ---------------------------------------------------------------- internals

    def _noise(self, rng: np.random.Generator, temperature: float) -> np.ndarray:
        std = math.sqrt(temperature)
        noise = rng.standard_normal((1, self.config.latent_dim)) * std
        if self.config.noise_clamp is not None:
            clamp = self.config.noise_clamp
            while True:
                outside = np.abs(noise) > clamp
                if not outside.any():
                    break
                noise[outside] = rng.standard_normal(int(outside.sum())) * std
        return noise.astype(np.float32)

    def _max_frames(self, token_count: int) -> int:
        config = self.config
        seconds = token_count / config.tokens_per_second_estimate + config.gen_seconds_padding
        return math.ceil(seconds * config.frame_rate)

    def _empty_flow_cache(self, capacity: int) -> _Cache:
        config = self.config
        return _Cache(
            (config.flow_layers, 2, 1, config.flow_heads, config.flow_head_dim), capacity
        )

    def _frontend(self, phonemes: bool) -> tuple[object, dict]:
        """The tokenizer and per-mode defaults for this call."""
        if not phonemes:
            return self.tokenizer, {}
        if self.phonemes_tokenizer is None:
            raise RuntimeError(
                f"{self.path.name} carries no phoneme adapter; export with "
                "--lora <checkpoint.pt> to enable phonemes=True"
            )
        return self.phonemes_tokenizer, dict(self.config.lora.get("defaults") or {})

    def _prefilled_voice(self, voice: str | np.ndarray | None, gate: float) -> tuple[_Cache, int]:
        """Flow-LM cache holding just the voice prompt; recomputed once per voice.

        The adapter changes the attention weights, so a prefilled voice belongs
        to the gate it was computed under. `voice=None` prompts with nothing,
        which is what an adapter trained without voice prompts expects.
        """
        if voice is None:
            return self._empty_flow_cache(256), 0
        key = (voice, gate) if isinstance(voice, str) else None
        if key is not None and key in self._voice_cache:
            values, length = self._voice_cache[key]
        else:
            cond = self.voice_conditioning(voice)
            cache = self._empty_flow_cache(cond.shape[1] + 1)
            outputs = self._run(
                cond=cond,
                gates=_COND,
                seq=cond.shape[1],
                noise=np.zeros((1, self.config.latent_dim), dtype=np.float32),
                flow_kv=cache,
                mimi_kv=_Cache(
                    (
                        self.config.mimi_layers,
                        2,
                        1,
                        self.config.mimi_heads,
                        self.config.mimi_head_dim,
                    ),
                    self.config.mimi_kv_len,
                ),
                mimi_offset=np.zeros((), dtype=np.int64),
                mimi_conv=np.zeros(self.config.conv_state_size, dtype=np.float32),
                lora=gate,
            )
            values, length = outputs["flow_kv_new"], cond.shape[1]
            if key is not None:
                self._voice_cache[key] = (values, length)
        cache = self._empty_flow_cache(length + 256)
        cache.append(values)
        return cache, length

    def _run(
        self,
        *,
        gates: np.ndarray,
        seq: int,
        noise: np.ndarray,
        flow_kv: _Cache,
        mimi_kv: _Cache,
        mimi_offset: np.ndarray,
        mimi_conv: np.ndarray,
        tokens: np.ndarray | None = None,
        latent: np.ndarray | None = None,
        is_bos: np.ndarray | None = None,
        cond: np.ndarray | None = None,
        decode_steps: float = 1.0,
        lora: float = 0.0,
    ) -> dict[str, np.ndarray]:
        config = self.config
        zeros = lambda *shape: np.zeros(shape, dtype=np.float32)  # noqa: E731
        feeds = {
            "tokens": tokens if tokens is not None else np.zeros((1, seq), dtype=np.int64),
            "latent": latent if latent is not None else zeros(1, seq, config.latent_dim),
            "is_bos": is_bos if is_bos is not None else zeros(1, seq, 1),
            "cond": cond if cond is not None else zeros(1, seq, config.model_dim),
            "gates": gates,
            "noise": noise,
            "flow_kv": flow_kv.window(),
            "flow_offset": np.asarray(flow_kv.length, dtype=np.int64),
            "mimi_kv": mimi_kv.window(config.mimi_kv_len),
            "mimi_offset": mimi_offset,
            "mimi_conv": mimi_conv,
            "decode_steps": np.asarray(decode_steps, dtype=np.float32),
        }
        feeds["lora"] = np.asarray(lora, dtype=np.float32)
        # Older exports lack the adapter and decode-step inputs; feed what exists.
        feeds = {name: value for name, value in feeds.items() if name in self._input_names}
        return dict(zip(self._output_names, self.session.run(None, feeds)))

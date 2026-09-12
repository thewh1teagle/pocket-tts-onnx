<div align="center">

# pocket-tts-onnx

**Streaming text to speech in a single ONNX file. First audio in 80 ms on CPU.
No torch.**

[**Try the demo**](https://huggingface.co/spaces/thewh1teagle/PocketTTS) ·
[**Models**](https://huggingface.co/thewh1teagle/pocket-tts-onnx) ·
[**TypeScript**](packages/pocket-tts-onnx/) ·
[**Docs**](docs/USAGE.md) ·
[**How it works**](docs/DESIGN.md)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://thewh1teagle-pockettts.static.hf.space/)
[![models](https://img.shields.io/github/v/release/thewh1teagle/pocket-tts-onnx?label=models)](https://github.com/thewh1teagle/pocket-tts-onnx/releases)
[![python](https://img.shields.io/badge/python-3.13%2B-blue)](pyproject.toml)
[![runtime](https://img.shields.io/badge/runtime-onnxruntime-005CED)](https://onnxruntime.ai)

</div>

---

[Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts) exported to one
self-contained `.onnx`. The graph, tokenizer, voices, encoder and adapter all
live in the same file, decoding frame by frame on onnxruntime alone.

## Install

```bash
uv add git+https://github.com/thewh1teagle/pocket-tts-onnx
```

## Use

```python
from pocket_tts_onnx import PocketTTS

tts = PocketTTS.from_pretrained("english")  # fetched once, then cached
samples, sample_rate = tts.create("Hello world.", voice="alba")
```

Audio starts before the sentence is finished:

```python
for frame in tts.stream("Hello world.", voice="alba"):
    play(frame)  # 80 ms of audio, ~20 ms after asking
```

`"english-ipa"` is the model with the Hebrew adapter bundled in. In TypeScript
it is `npm install pocket-tts-onnx`, same model, same streaming, in a browser
tab or in Node; see [`packages/pocket-tts-onnx`](packages/pocket-tts-onnx/).

## Docs

* [Examples](examples/): writing a wav, streaming to your speakers, Hebrew,
  tuning a take, changing its speed
* [Using it](docs/USAGE.md): loading models, voices, cloning, phonemes, adapters,
  decode steps
* [How it works](docs/DESIGN.md): the streaming graph, what rides in the file,
  int8, numbers, and what was measured against upstream
* [Exporting](docs/EXPORT.md): building your own `.onnx`
* [The web demo](docs/WEB.md): running it all in the browser

## Voice cloning

Clone only your own voice, or one you have explicit permission to use. Do not
use it to impersonate anyone, to mislead, or in ways that break the law where
you are. The software is provided as is; what you generate with it is your
responsibility.

## License

[CC BY 4.0](LICENSE)

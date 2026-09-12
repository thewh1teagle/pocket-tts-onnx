# Exporting

```bash
uv run pocket-tts-onnx-export pocket-tts-english.onnx --quantize
uv run pocket-tts-onnx-export pocket-tts-german.onnx --language german_24l --voices all
uv run pocket-tts-onnx-export pocket-tts-ipa.onnx --lora adapter/latest.pt
uv run pocket-tts-onnx-export pocket-tts-fast.onnx --max-decode-steps 1
```

Export needs torch and `pocket-tts`, both development dependencies pulled in by
`uv sync`. The runtime needs neither.

| flag | what |
| --- | --- |
| `--language` | which upstream config to build (default `english`) |
| `--voices` | voice names or wav paths to bake in, or `all` (default: six English voices) |
| `--voice-seconds` | how much of each voice prompt to encode (default 20) |
| `--lora` | a finetune checkpoint (`.pt`) to bundle as an adapter |
| `--quantize` | int8 weights for the flow LM transformer |
| `--max-decode-steps` | largest flow decode count the graph can serve (default 4) |

## Naming a voice's language

A voice can be prefixed with the language it was recorded in:

```bash
--voices en:alba en:michael he:omer.wav he:liat.wav
```

That lands in the metadata as `voice_languages`, and the web app uses it to show
only the voices that speak the language you are typing. Voices without a prefix
are shown everywhere.

## Why the pipeline is rewritten

The upstream modules keep streaming state in mutable dicts written in place, and
branch on Python values such as `offset.item()`. Neither survives tracing, so
`src/pocket_tts_onnx/export/step_model.py` re-expresses the whole per-step
pipeline as a pure `state in -> state out` function over the upstream weights:
the same `nn.Linear` and `nn.Conv1d` objects, only the forward logic is new.

## Two things the exporter has to fix up

The TorchScript exporter leaves `initializer -> Identity -> consumer` chains
behind, and onnxruntime's own identity elimination then trips over them and
refuses to load the model. `fold_identity` rewires them away.

The quantizer rewrites `Gemm` into `MatMul` by transposing each weight **in
place**, so a weight shared across the unrolled flow head is transposed once per
copy and comes out wrong on alternate nodes. `gemm_to_matmul` does that
conversion first, transposing each weight exactly once.

## Adding an adapter

A finetune checkpoint is expected to carry `lora`, `extra_embed`, `ipa_chars`,
`out_norm` and `out_eos`; anything else in it, such as optimizer state, is
ignored. The exporter refuses a checkpoint missing any of those.

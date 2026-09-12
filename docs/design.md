# How it works

## One file

A `.onnx` holds everything the runtime needs:

* the graph for one streaming step,
* the model geometry, sampling defaults and text-preparation rules
  (`pocket_tts_config` in the ONNX metadata),
* the sentencepiece tokenizer (`pocket_tts_tokenizer`),
* the precomputed voice conditionings (`pocket_tts_voice/<name>`),
* the mimi encoder graph, for cloning a voice (`pocket_tts_encoder`),
* an optional finetune adapter.

## Streaming

The graph is one step, not one utterance. Each call takes the keys and values
and the mimi convolution state produced so far, and returns one latent and one
80 ms audio frame, so audio leaves the model while the rest of the sentence is
still being generated.

Only the keys and values computed *by that step* come back out. The runtime
appends them to its own buffer and hands the next step a contiguous window of
the past, so no cache is ever copied in and out whole.

The same graph serves three roles, chosen by the `gates` input: embed a text
prompt, take the voice conditioning, or consume the previous latent. A prefill
call runs the whole pipeline anyway and the caller simply keeps the flow-LM
cache and drops the audio, which costs one wasted mimi decode per prefill and
saves a second graph.

Long text is still split into sentence chunks: the model is trained on single
sentences and caps at 50 tokens. Each chunk restarts from the voice state, and
streams frame by frame like everything else.

## Adapters

Nothing about the graph changes when one is bundled. A rank-16 adapter on the
attention projections, plus the retrained final norm and EOS head, ride along as
deltas multiplied by a `lora` scalar input, and the embedding table simply gains
rows that plain text never reaches. So `lora=0` is the base model bit for bit,
`lora=1` is the adapted one, and the cost when it is off is nothing. The Hebrew
IPA adapter adds 0.73 M parameters, or 3 MB.

## int8

`--quantize` roughly halves the file and speeds it up:

| | float32 | int8 |
| --- | --- | --- |
| file | 457 MB | 231 MB |
| RTF | 8.5 to 9.1x | 10.4 to 10.6x |

Only the flow LM transformer's attention and feed-forward matmuls are quantized,
24 of 174. That is the set upstream quantizes and measured no WER change on
(`pocket_tts/quantization.py`). The flow head, mimi, the embeddings, the adapter
and every convolution stay float32. They are found by weight shape rather than by
name, and the exporter refuses to continue unless it finds exactly four per
layer.

Judge the result by ear, not by SNR against the float32 output. The flow sampler
is chaotic, so any perturbation gives a different but equally valid waveform:
whole-utterance SNR lands near 0 dB even when nothing is wrong, and two float32
implementations of the same model only reach 29 dB. What is worth reading is the
first frame, before the loop amplifies anything (34 to 46 dB), and whether EOS
still fires on the same frame, which it does.

## What was measured against upstream

With sampling noise held at zero, so the two paths are comparable:

| | first frame | notes |
| --- | --- | --- |
| base English | 1.5e-05 | vs `TTSModel.generate_audio` |
| Hebrew IPA adapter | 4.3e-08 | vs the adapter's `finetune/infer.py`, 57 dB overall |
| adapter gate off | 0.0 | bit-identical to an export without the adapter |
| voice cloning | 1.2e-04 | vs torch, below the fp16 step used to store voices |

Decode steps 1, 2 and 4 each match their torch equivalent, and durations agree.

## Layout

| path | what |
| --- | --- |
| `src/pocket_tts_onnx/tts.py` | the runtime: `PocketTTS`, `.create()`, `.stream()` |
| `src/pocket_tts_onnx/text.py` | sentence splitting, prompt normalisation, the mixed IPA tokenizer |
| `src/pocket_tts_onnx/g2p.py` | text to IPA, via espeak and conikud |
| `src/pocket_tts_onnx/audio.py` | reading and resampling a voice prompt |
| `src/pocket_tts_onnx/export/step_model.py` | ONNX-exportable rewrite of inference |
| `src/pocket_tts_onnx/export/export.py` | the exporter and its CLI |

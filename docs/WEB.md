# The web demo

`web/` is a React app that runs the whole model in the browser on
onnxruntime-web. Nothing is uploaded: the text, the voice you drop in and the
audio all stay in the tab.

Live at **https://thewh1teagle-pockettts.static.hf.space/**, and mirrored to
https://thewh1teagle.github.io/pocket-tts-onnx/

That is the Space's own direct URL, the one its "Embed this Space" dialog
offers, and every link here points at it rather than at the
https://huggingface.co/spaces/thewh1teagle/PocketTTS page. The page is a
wrapper that loads the same origin in an iframe, and a framed document can only
be cross-origin isolated if its embedder is isolated and hands it the
`cross-origin-isolated` permission; huggingface.co does neither. So inside the
wrapper there is no `SharedArrayBuffer`, onnxruntime gets one thread, and the
demo runs at about 2.2x real time where the direct URL runs at 5.5x. Nothing
in `coi.js` can lift that, because COOP is a header only a top-level document
is asked about.

## Running it

```bash
cd web
npm install
npm run dev
```

In development the app serves models from `web/models/`; build that asset set
with the exporter:

```bash
uv run pocket-tts-onnx-export pocket-tts-web.onnx --quantize \
    --lora adapter/latest.pt --voices he:omer.wav he:liat.wav en:alba en:anna en:michael
uv run pocket-tts-onnx-web pocket-tts-web.onnx web/models/en
cp renikud.onnx web/models/en/
cp -r web/models/en web/models/he
```

In production it fetches them from Hugging Face. Point it anywhere else with
`?models=<url>` or by setting `VITE_MODELS_URL` at build time.

## Everything runs in a worker

onnxruntime-web executes its wasm synchronously on whichever thread calls it, so
running the model on the page blocked React and starved the audio clock. The
symptom was a status line frozen mid-generation on a fast machine, and playback
arriving syllable by syllable on a slow one. The model, both phonemizers and the
voice encoder now live in the package's `worker.ts`; the page only ever receives
finished frames.

That fixes the freeze but not slow hardware, so the player rebuffers rather than
stutters. It banks about 0.7 seconds before it starts, plays that out, and if
the lead runs down it holds the next frames and banks again: one clean pause,
the way a video player pauses, instead of a gap between every 80 ms frame. It
always streams; it never waits for the whole take.

## Input formats

The Hebrew tab takes ordinary text and decides per part what to do with it:
double brackets are passed through as typed, IPA or nikud alike, Hebrew with
nikud is kept as typed (plain or with the phonikud marks for stress, vocal shva
and prefix boundaries), unvocalized Hebrew goes through renikud, and Latin words
go through espeak. There is no separate phonemes tab, because `[[ʃalˈom]]` anywhere in
the line does that job, in either language.

espeak is an 18 MB wasm build, so it is fetched only when a line actually mixes
scripts, the same treatment renikud and the voice encoder get. Writing plain
Hebrew never pays for it. Its output matches the Python package's espeak
exactly, once the zero-width joiners `--ipa=3` puts inside diphthongs are
stripped and the punctuation it drops is put back.

## Cloning a voice

Two ways in, both local: drop an audio file, or press Record and speak for five
to ten seconds. Either way the audio is decoded in the page, resampled to
24 kHz, and encoded once into conditioning the model reuses, so the encoder
never runs again during synthesis. The recording is never written anywhere and never
sent anywhere; the microphone is released the moment you stop.

Streaming keeps every frame it has played rather than letting it go, so the
take can be scrubbed while it is still being generated: dragging the playhead
drops what was scheduled and schedules the tail again from the new point, and
frames still to come land after it as usual. The finished take is then handed
to a small player for replay, seeking and download.

The waveform is drawn from the frames themselves, one bar per 80 ms frame,
filling left to right as the model produces them. The same drawing becomes the
finished take's scrubber, so nothing jumps when generation ends. It follows the
pointer through a drag either way; the finished take, which the browser seeks
cheaply, moves with it, while a take still generating moves on release.
Playback can be paused mid-stream while frames keep arriving, and the streaming
view is held until the last frame has actually been heard rather than when the
model stops producing them.

Three things open over the page rather than sitting under the composer: the
voice, the code, and the tokens. Picking a voice is a moment's work and cloning
one is rarer still, so the voice name in the header is the way in to both, and
what is left on screen is the text and the take.

The token view shows the text after phonemization and the tokens it became, with
the adapter's atomic characters marked apart from the SentencePiece pieces. It is
the quickest way to see why a line came out the way it did. Tokens are collected
on every run — one extra tokenize of a sentence, next to a phonemizer and a voice
warmup — so the answer is there whenever the view is opened. It holds the
previous run's tokens, dimmed, while the next run's are on their way, rather than
emptying and refilling in that moment.

The code view is the same job in Python: install, generate, stream, with the
generate snippet following the language selected in the composer. Nothing in it
tracks what you have typed, because a snippet that changes as you type is a toy
rather than something to copy. Prism highlights it, with one token added to each
of its grammars: bash only knows the classic unix commands, so `uv` needs
teaching, and its python `function` token covers `def` sites but not calls.

## The asset set

The single-file layout is right for a Python process that opens the model once.
A browser wants the opposite, the smallest possible first download, so
`pocket-tts-onnx-web` splits it:

| file | size | when |
| --- | --- | --- |
| `model.onnx` | 177 MB | on load |
| `assets.json` | 1.4 MB | on load, alongside the model |
| `renikud.onnx` | 21 MB | first time Hebrew is used |
| `encoder.onnx` | 39 MB | first time a voice is cloned |

Stripping the metadata and the encoder out of the graph takes about 55 MB off
what a visitor waits for. Downloads stream so the progress bar is real, and land
in the Cache API, so a second visit pays no network at all.

## One model per language

Every language is an asset set in a folder named by its code (`en/`, `he/`,
`es/`, `fr/`, …). English and Hebrew are the same bytes published twice,
because Hebrew is an adapter on the English weights; the page compares digests,
so switching between them fetches nothing and keeps the loaded model. Spanish,
French, German, Italian and Portuguese are separate models upstream, built with
`chore export-languages`. Choosing one of those languages asks before fetching it,
with the size from its manifest, and the model is cached like the first; going
back to a language whose model is already cached reloads it without asking.
French only exists as a 24-layer model, so it is about twice the size of the
others.

## Why Hugging Face and not the GitHub release

GitHub serves release assets **without CORS headers**, so a browser cannot fetch
them at all: `fetch` throws before it sees a byte. The release is still the
right home for the single-file Python models, which are downloaded by tools
rather than pages. The web asset set lives at
[thewh1teagle/pocket-tts-onnx](https://huggingface.co/thewh1teagle/pocket-tts-onnx),
which serves `access-control-allow-origin: *`.

## What had to be ported

The runtime is Python-free, so everything the package does before and after the
graph had to be rewritten in TypeScript, and each piece was checked against the
Python it came from:

All of it lives in `packages/pocket-tts-onnx/src`, which is published to npm as
[`pocket-tts-onnx`](../packages/pocket-tts-onnx/) — the site is a consumer of
the package like any other.

| file | what | checked against Python |
| --- | --- | --- |
| `sentencepiece.ts` | unigram tokenizer, byte fallback | 56 cases, exact |
| `text.ts` | chunking, prompt prep, mixed IPA tokenizer | every case, exact |
| `g2p.ts` | renikud Hebrew G2P | 7 sentences, exact |
| `mixed.ts` | routing each part of the text to the right phonemizer | shares its cases with Python |
| `espeak.ts` | English phonemes, from espeak-ng in wasm | same IPA as Python's espeak |
| `worker.ts` | all of the above, off the main thread | — |
| `tts.ts` | the streaming loop and its caches | see below |
| `recorder.ts` | microphone capture and its level meter | — |
| `onnxMeta.ts` | ONNX metadata without parsing the graph | — |

`onnxMeta.ts` exists because onnxruntime-web exposes input and output names
but not `metadata_props`, and renikud keeps its vocabularies there. Walking the
top level of the protobuf is cheap: every field is length-prefixed, so the
multi-megabyte graph is one skip.

## On numbers not matching exactly

Audio from the browser is not sample-identical to audio from Python, and cannot
be. Feeding one identical step through both, onnxruntime's wasm kernels and its
native CPU kernels differ by about 2e-3 on the EOS logit and 3e-4 on the audio,
which is ordinary floating-point disagreement between two implementations. The
flow sampler is chaotic, so that difference grows into a different-but-equally-valid
take. Everything upstream of the graph (tokens, chunk boundaries, phonemes,
voice conditioning) was verified byte-identical, which is the part a bug would
show up in.

## Why not WebGPU

onnxruntime-web has a WebGPU execution provider, and this model does load on it.
Measured here on the same graph with the same inputs, a generation step took
**31 ms on wasm against 38 ms on WebGPU**, slower rather than faster. The model
is int8, which the GPU path does not favour, and it runs one 80 ms frame at a
time,
so dispatch overhead outweighs the arithmetic. It would also mean shipping the
27 MB jsep build rather than the 14 MB one. A float32 export might suit the GPU
better, at four hundred megabytes of download.

## Speed

Roughly 6× real time in Chrome on an M-series laptop, with audio starting
**under 200 ms** after the click: 70 ms for English, 130 ms for Hebrew, which
also has to phonemize. Two things buy that.

**Threads.** onnxruntime's wasm build is threaded, but shared memory needs the
page to be cross-origin isolated, and GitHub Pages sends no headers at all. The
development server sends `Cross-Origin-Opener-Policy` and
`Cross-Origin-Embedder-Policy: credentialless` itself; the built site gets there
through `public/coi.js`, a service worker that adds those headers to every
response once it controls the page. `credentialless` rather than
`require-corp`, because the models come from Hugging Face, which sends no CORP
header of its own. Where isolation fails the runtime falls back to one thread
and everything still works, slower. Brave answers `hardwareConcurrency` with
two whatever the machine is, so its answer is ignored.

**Warming the voice.** A voice prompt is half a second of attention before a
single frame can come out, and it is the same half second whenever it happens.
`App.tsx` asks the worker to do it the moment a voice or a language is chosen,
so the take itself only ever pays for its own text. The Hebrew G2P and espeak
are warmed the same way, which is why the first Hebrew take costs what the
tenth does.

# pocket-tts-onnx

**Streaming text to speech in TypeScript, on onnxruntime. In a browser tab or in
a script, with nothing uploaded either way.**

[Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts) exported to one
self-contained `.onnx`. The graph, tokenizer, voices, encoder and adapter all
live in the same file, decoding frame by frame — around 6× real time in a
browser and 10× in Node, with the first 80 ms of audio out before the sentence
is finished.

```bash
npm install pocket-tts-onnx
```

```ts
import { encodeWav, load } from "pocket-tts-onnx";

const tts = await load({ language: "english" });
const audio = await tts.speak("Hello there.");
// mono float32 in [-1, 1], at tts.sampleRate
```

Seven languages — `english`, `hebrew`, `spanish`, `french`, `german`, `italian`,
`portuguese` — each a folder of weights fetched the first time you ask for it.

## The models are not in the package

The weights are 177 MB and stay out of npm. They are fetched on first use from
[Hugging Face](https://huggingface.co/thewh1teagle/pocket-tts-onnx) and kept
afterwards: in the Cache API in a browser, under `~/.cache/pocket-tts-onnx` in
Node. Every run after the first starts speaking straight away.

| | | |
| --- | --- | --- |
| `model.onnx` | 177 MB | always |
| `assets.json` | 1.6 MB | always — the tokenizer and the voices |
| `renikud.onnx` | 21 MB | Hebrew only, on first Hebrew line |
| `espeak-ng.wasm` | 18 MB | only for Latin words inside Hebrew |
| `encoder.onnx` | 39 MB | only if you clone a voice |

Point `modelsUrl` somewhere else to host them yourself, or at a folder on disk
(Node only) to skip the network entirely:

```ts
await load({ modelsUrl: "https://example.com/models/en/" });
await load({ modelsUrl: "./models/en/" });
```

## In Node

```bash
cd packages/pocket-tts-onnx
npm install                      # `prepare` builds the package
pnpx tsx examples/english.ts
pnpx tsx examples/hebrew.ts "שלום, מה שלומך?"
```

Both write a `.wav` next to themselves and are commented with what they fetch
and why — [`examples/english.ts`](examples/english.ts) and
[`examples/hebrew.ts`](examples/hebrew.ts) are the shortest way to see the whole
API in use.

Node runs on `onnxruntime-node`, an optional dependency: the native binding is
roughly ten times faster than wasm here, which is the difference between a
script that finishes while you watch and one you walk away from. It installs by
default; if its build fails on your platform, the browser path still works.

## In a browser

The main entry point works on a page as it stands, but it decodes on whatever
thread calls it, and onnxruntime's wasm runs synchronously — enough to freeze
React and starve the audio clock. `pocket-tts-onnx/browser` puts a worker in
front, so the page only ever receives finished frames:

```ts
import { Engine, FramePlayer } from "pocket-tts-onnx/browser";
import PocketTTSWorker from "pocket-tts-onnx/worker?worker";  // Vite

const engine = await Engine.load({
  language: "english",
  worker: () => new PocketTTSWorker(),
  onProgress: (stage, { loaded, total }) => console.log(stage, loaded / total),
});

const player = new FramePlayer(engine.sampleRate);
await player.start();  // from a click; browsers will not start audio otherwise
for await (const frame of engine.speak("Hello there.", "alba")) player.push(frame);
```

Hand `Engine.load` a worker rather than letting it build its own. Left to
itself it uses `new Worker(new URL("./worker.js", import.meta.url))`, which is
right for a plain module server but which most bundlers cannot follow into a
dependency — Vite inlines the file as a data URL and loses its imports.

`pocket-tts-onnx/browser` also carries `decodeAudioFile`, `resample`,
`Recorder`, `encodeWavBlob` and the loudness helpers the demo site uses.

### Two files to serve yourself

onnxruntime and espeak both fetch their own wasm at runtime. The defaults are a
pinned copy on jsdelivr for espeak and onnxruntime's own, which is right until
you would rather not depend on a third party:

```ts
await Engine.load({
  ortWasmUrl: "/ort/",              // where you copied onnxruntime-web/dist/*.wasm
  espeakWasmUrl: "/espeak-ng.wasm", // espeak-ng/dist/espeak-ng.wasm
});
```

Threads need `SharedArrayBuffer`, which needs the page cross-origin isolated
(`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy:
credentialless`). Without isolation it runs single-threaded on SIMD, which
works and is slower.

## Hebrew

Hebrew is an adapter on the English weights, and it reads stressed IPA rather
than spelling, so text is phonemized first. That happens by itself — a line
with Hebrew in it takes the adapter, a line without stays on the base model,
which reads plain English better than phonemes would.

Mixed lines work: Hebrew goes through [renikud](https://github.com/thewh1teagle/renikud),
Latin words through espeak, and words already carrying nikud are kept as
written. Double brackets pass IPA straight through, which is how you fix a word
the phonemizer gets wrong:

```ts
await tts.speak("שלום, אני מריץ [[pˈɑkət]] TTS");
```

Numbers are read before any of that. renikud has no consonant for a `2`, so
`₪25` would be spelled out character by character; instead numbers, money,
dates, times and units become the words a person would say, and `[[literals]]`,
nikud, Latin runs, URLs and emails are left as written:

```ts
await tts.speak("הכרטיס עלה ₪25 בשעה 14:30");
// עשרים וחמישה שקלים בשעה שתיים וחצי אחר הצהריים

await Engine.load({ hebrewNormalization: { clock: "24" } });  // ארבע עשרה שלושים
await Engine.load({ hebrewNormalization: false });            // as written
```

Its 1.3 MB of wasm is fetched the first time a Hebrew line needs it, so an
English page never pays for it.

## Voice cloning

```ts
const voice = await tts.clone(samples);   // mono float32 at tts.sampleRate
await tts.speak("Now in that voice.", { voice });
```

Twenty seconds is as much as the encoder reads. Clone only your own voice, or
one you have explicit permission to use.

## API

| | |
| --- | --- |
| `load(options)` | fetch a language and return a `Pipeline` |
| `pipeline.speak(text, options)` | the whole utterance as one `Float32Array` |
| `pipeline.stream(text, options)` | the same audio, 80 ms at a time |
| `pipeline.clone(samples)` | conditioning to pass back as `voice` |
| `pipeline.prepare(voice)` | warm a voice before anyone is waiting |
| `pipeline.voices` / `.sampleRate` / `.defaultVoice` | what this model carries |
| `encodeWav(samples, rate)` | 16-bit PCM wav, as bytes |
| `clearAssetCache()` | drop every cached model |

`speak` and `stream` take `voice`, `temperature`, `decodeSteps`, `seed`,
`signal`, and `onStatus` / `onProgress` / `onDebug` callbacks. A `seed` makes a
take reproducible.

`PocketTTS` is exported too, for handing the decoder your own model bytes and
skipping the fetching and phonemizing above it.

## Requirements

Node 20+, or any browser with WebAssembly. Types are included.

## License

[CC BY 4.0](LICENSE)

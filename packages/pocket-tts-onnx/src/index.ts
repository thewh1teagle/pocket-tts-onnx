/**
 * pocket-tts, on onnxruntime, in a page or in a script.
 *
 * Everything exported here runs in both: the model, the phonemizers and the
 * decode loop touch nothing but typed arrays. `pocket-tts-onnx/browser` adds
 * the parts that need a DOM — a worker, playback, recording.
 *
 *     import { load } from "pocket-tts-onnx";
 *     const tts = await load({ language: "english" });
 *     const audio = await tts.speak("Hello there.");
 */

export { Pipeline, load } from "./pipeline.js";
export type {
  DebugPayload,
  Manifest,
  ProgressHandler,
  SpeakOptions,
  Stage,
} from "./pipeline.js";

export { MODELS_URL, resolveOptions } from "./options.js";
export type { Options } from "./options.js";

export { AVAILABLE, LANGUAGES, language } from "./languages.js";
export type { Language, Mode } from "./languages.js";

export { encodeWav } from "./wav.js";
export { clearAssetCache, fetchAsset, fetchJson, isCached } from "./assets.js";
export type { Progress } from "./assets.js";

// The layer underneath, for anyone who wants to hand it their own bytes.
export { PocketTTS } from "./tts.js";
export type { Assets, ModelConfig } from "./tts.js";
export { HebrewG2P } from "./g2p.js";
export { loadEspeak, phonemizeEnglish, espeakReady } from "./espeak.js";
export { phonemizeMixed } from "./mixed.js";
export type { Phonemizers } from "./mixed.js";
export { normalizeHebrew, normalizerReady, prepareNormalizer } from "./normalize.js";
export type { HebrewNormalization, NormalizerConfig } from "./normalize.js";
export { breakParagraphs } from "./text.js";

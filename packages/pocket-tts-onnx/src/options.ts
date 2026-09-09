/**
 * What a caller may say, and what the rest of the package reads.
 *
 * Everything environment-shaped is settled here and nowhere else, so no module
 * further in has to know whether it is running in a page or in a script, or
 * reach for a bundler's `import.meta.env`. Every field has a default that
 * works, which is what makes `await load()` a complete program.
 */

import { configureRuntime } from "#platform";

import { language as languageByName } from "./languages.js";
import { normalizerConfig, type HebrewNormalization, type NormalizerConfig } from "./normalize.js";

/**
 * Where the published models live.
 *
 * Hugging Face rather than the GitHub release: release assets are served
 * without CORS headers, so a browser cannot fetch them at all. Each language
 * is a folder of its own underneath.
 */
export const MODELS_URL = "https://huggingface.co/thewh1teagle/pocket-tts-onnx/resolve/main/";

export interface Options {
  /**
   * Which language to speak, by name: `english`, `hebrew`, `spanish`,
   * `french`, `german`, `italian`, `portuguese`. It only picks the folder
   * under `modelsUrl`, so `modelsUrl` on its own overrides it entirely.
   */
  language?: string;
  /**
   * The folder holding `manifest.json` and the weights. An `https:` URL, or a
   * path or `file:` URL when running in Node. Defaults to this language's
   * folder on Hugging Face.
   */
  modelsUrl?: string;
  /**
   * Where onnxruntime's own wasm sits, for a page that serves it from
   * somewhere other than the runtime's default. Ignored in Node.
   */
  ortWasmUrl?: string;
  /** espeak's wasm, needed only for Latin words inside Hebrew text. */
  espeakWasmUrl?: string;
  /** Threads onnxruntime may use. Ignored in Node, which uses the CPU provider. */
  threads?: number;
  /**
   * Read Hebrew numbers, money, dates, times and units as words before the g2p
   * sees them, so `₪25` is spoken rather than spelled. On by default; pass
   * `false` to send the text through as written, or an object to set the
   * reading style — 12- or 24-hour clock, date order, and the rest.
   */
  hebrewNormalization?: HebrewNormalization;
}

export interface Resolved {
  baseUrl: string;
  espeakWasmUrl?: string;
  /** The reading style for Hebrew normalization, or `null` when it is off. */
  hebrewNormalization: NormalizerConfig | null;
}

const slash = (url: string) => (url.endsWith("/") ? url : `${url}/`);

/** Settle a caller's options, and tell onnxruntime what it needs to know. */
export function resolveOptions(options: Options = {}): Resolved {
  configureRuntime({ ortWasmUrl: options.ortWasmUrl, threads: options.threads });
  // `||`, not `??`: an unset environment variable reaches a build as an empty
  // string, which would otherwise be taken as a real base and 404.
  const base = options.modelsUrl || MODELS_URL + languageByName(options.language ?? "english").model;
  return {
    baseUrl: slash(base),
    espeakWasmUrl: options.espeakWasmUrl,
    hebrewNormalization: normalizerConfig(options.hebrewNormalization),
  };
}

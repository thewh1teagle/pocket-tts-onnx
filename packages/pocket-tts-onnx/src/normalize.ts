/**
 * Hebrew text normalization, in front of the g2p.
 *
 * renikud reads letters, not digits: `₪25` reaches it as three characters it
 * has no consonant for, and comes out the other side as literal `₪25` for the
 * adapter's tokenizer to make what it can of. So numbers, money, dates, times
 * and units are turned into the words a person would say before any of that
 * happens — `heb-tts-normalizer` does exactly that job and leaves everything
 * this pipeline cares about alone: `[[literals]]`, nikud, the phonikud `|`
 * prefix, Latin runs, URLs and emails all come back as written.
 *
 * The wasm is 1.3 MB and only Hebrew needs it, so it is imported the first time
 * a Hebrew line asks and never in an English page's bundle.
 */

import type { Config } from "heb-tts-normalizer";

export type { Config as NormalizerConfig };

/** `false` turns normalization off; an object is the reading style to use. */
export type HebrewNormalization = boolean | Config;

type Module = typeof import("heb-tts-normalizer");

let pending: Promise<Module> | null = null;

async function load(): Promise<Module> {
  pending ??= (async () => {
    const module = await import("heb-tts-normalizer");
    await module.init();
    return module;
  })().catch((error) => {
    // A failed load must not poison the module: let the next line try again.
    pending = null;
    throw error;
  });
  return pending;
}

/** Whether the normalizer is loaded and `normalizeHebrew` will not have to wait. */
export function normalizerReady(): boolean {
  return pending !== null;
}

/** The config a setting stands for, or `null` when it is off. */
export function normalizerConfig(setting: HebrewNormalization | undefined): Config | null {
  if (setting === false) return null;
  return setting === true || setting === undefined ? {} : setting;
}

/** Everyday Hebrew in, speakable Hebrew words out. */
export async function normalizeHebrew(text: string, config: Config = {}): Promise<string> {
  const { normalize } = await load();
  return normalize(text, config);
}

/** Load the wasm now, so the first Hebrew take does not pay for it. */
export async function prepareNormalizer(): Promise<void> {
  await load();
}

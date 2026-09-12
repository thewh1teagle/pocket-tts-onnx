"""Grapheme-to-phoneme, for models that read IPA rather than spelling.

An adapter trained on phonemes needs IPA in, so this gets there from ordinary
text:

    from pocket_tts_onnx import phonemize

    tts.create(phonemize("How are you today?"), voice=cond, phonemes=True)
    tts.create(phonemize("שלום עולם", language="he"), voice=cond, phonemes=True)

English goes through espeak (via phonemizer, with the library bundled by
espeakng-loader). Hebrew has no espeak path worth using, so it goes through
conikud, a small ONNX G2P that fetches its own weights on first use.

Backends are built once and reused. Constructing either one — loading the espeak
shared library, or an onnxruntime session — costs far more than the
phonemization itself.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_LANGUAGE = "en-us"
# Text between double brackets is already unambiguous and is passed straight through.
LITERAL = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)
_WORDS = re.compile(r"(\s+)")
_SCRIPTS = re.compile(r"[\u0590-\u05FF]+|[^\u0590-\u05FF]+")
HEBREW_LANGUAGES = {"he", "he-il", "heb", "hebrew"}
# espeak wants a full tag, but "en" is what everyone reaches for.
LANGUAGE_ALIASES = {"en": "en-us", "english": "en-us"}
CONIKUD_ENV = "CONIKUD_MODEL"


@lru_cache(maxsize=None)
def _espeak_ready() -> None:
    from espeakng_loader import get_data_path, get_library_path, make_library_available
    from phonemizer.backend import EspeakBackend

    make_library_available()
    os.environ.setdefault("ESPEAK_DATA_PATH", str(get_data_path()))
    EspeakBackend.set_library(get_library_path())


@lru_cache(maxsize=8)
def _espeak_backend(language: str):
    from phonemizer.backend import EspeakBackend

    _espeak_ready()
    return EspeakBackend(language, preserve_punctuation=True, with_stress=True)


@lru_cache(maxsize=4)
def _conikud(model: str | None):
    """The Hebrew G2P: an explicit path, else `$CONIKUD_MODEL`, else the Hub copy."""
    try:
        from conikud_onnx import G2P
    except ImportError:
        raise RuntimeError(
            "Hebrew G2P needs conikud-onnx: pip install git+https://github.com/conikud/conikud-onnx"
        ) from None
    return G2P(model or os.environ.get(CONIKUD_ENV) or None)


def phonemize(
    text: str,
    language: str = DEFAULT_LANGUAGE,
    model: str | Path | None = None,
    normalize: bool = False,
) -> str:
    """Stressed IPA for `text`, punctuation preserved.

    ```python
    phonemize("How are you today?")        # 'hˌaʊ ɑːɹ juː tədˈeɪ?'
    phonemize("שלום עולם", language="he")  # 'ʃlˈom ʔolˈam'
    ```

    `model` points at a local conikud export for Hebrew; without it the weights
    are taken from `$CONIKUD_MODEL` or fetched from the Hub and cached.
    `normalize=True` speaks Hebrew numbers, money, dates and times as words.
    """
    return phonemize_all([text], language, model, normalize)[0]


def phonemize_all(
    texts: list[str],
    language: str = DEFAULT_LANGUAGE,
    model: str | Path | None = None,
    normalize: bool = False,
) -> list[str]:
    """`phonemize` over a list; English does the whole list in one espeak call."""
    language = language.lower()
    if language in HEBREW_LANGUAGES:
        g2p = _conikud(str(model) if model is not None else None)
        return [g2p.phonemize(text, normalize=normalize).strip() for text in texts]
    language = LANGUAGE_ALIASES.get(language, language)
    return [line.strip() for line in _espeak_backend(language).phonemize(list(texts), strip=True)]


def _has_nikud(word: str) -> bool:
    """A nikud mark, or the phonikud prefix boundary, makes a word unambiguous."""
    return any(0x0590 <= ord(char) <= 0x05CF or char == "|" for char in word)


def _script(run: str) -> str:
    if any(0x05D0 <= ord(char) <= 0x05FF for char in run):
        return "hebrew"
    return "latin" if any(char.isalpha() for char in run) else "neutral"


def _runs(text: str) -> list[tuple[str, str]]:
    """Split into stretches of one script, keeping vocalized words whole."""
    out: list[tuple[str, str]] = []
    for token in _WORDS.split(text):
        if token == "":
            continue
        if token.isspace():
            out.append(("neutral", token))
        elif _has_nikud(token):
            out.append(("vocalized", token))
        else:
            # A word can hold both scripts, as in "ב-Google".
            for run in _SCRIPTS.findall(token):
                out.append((_script(run), run))
    return out


def _groups(text: str) -> list[tuple[str, str]]:
    """Merge neighbouring runs so each G2P sees whole phrases, not fragments."""
    groups: list[tuple[str, list[str]]] = []
    for kind, run in _runs(text):
        if groups and (kind == "neutral" or groups[-1][0] == kind):
            groups[-1][1].append(run)
        elif groups and groups[-1][0] == "neutral" and len(groups) == 1:
            groups[-1] = (kind, [*groups[-1][1], run])
        else:
            groups.append((kind, [run]))
    return [(kind, "".join(parts)) for kind, parts in groups]


def phonemize_mixed(
    text: str,
    model: str | Path | None = None,
    language: str | None = DEFAULT_LANGUAGE,
    normalize: bool = True,
) -> str:
    """Turn everyday mixed text into what a multiformat adapter expects.

    Each part goes the shortest way to phonemes it can:

    * `[[ʃalˈom]]` is already unambiguous, so the brackets come off and nothing else
      happens to it;
    * Hebrew carrying nikud is already unambiguous, so it is kept exactly as
      written and tokenized as atomic Hebrew and nikud characters;
    * unvocalized Hebrew goes through conikud, fetched on first use unless
      `model` points at a local export;
    * Latin script goes through espeak, so an English word inside a Hebrew
      sentence is spoken rather than spelled. Pass `language=None` to leave it
      as written instead.

    Hebrew is normalized on the way in, so `\u20aa25` is spoken rather than
    spelled: conikud reads letters, not digits, and numbers, money, dates and
    times become the words a person would say. Pass `normalize=False` to send
    the text through as written.

    ```python
    phonemize_mixed("אני עובד עם Photoshop כל יום")
    ```
    """
    out: list[str] = []
    at = 0
    for match in LITERAL.finditer(text):
        out.append(_phonemize_plain(text[at : match.start()], model, language, normalize))
        out.append(match.group(1))
        at = match.end()
    out.append(_phonemize_plain(text[at:], model, language, normalize))
    return _tidy("".join(out).strip())


def _tidy(ipa: str) -> str:
    """Clean up the seams between two phonemizers.

    A one-letter Hebrew prefix such as the `ב` of `ב-Google` reaches conikud
    with no word around it, and can come back as a bare stress mark that then
    collides with the stress of the word after it.
    """
    ipa = re.sub(r"\u02c8{2,}", "\u02c8", ipa)
    return re.sub(r"\u02c8(?=[\s,.!?;:]|$)", "", ipa)


def _phonemize_plain(
    text: str, model: str | Path | None, language: str | None, normalize: bool
) -> str:
    pieces = []
    for kind, group in _groups(text):
        if kind == "hebrew":
            pass
        elif kind == "latin" and language is not None:
            pass
        else:
            pieces.append(group)  # vocalized, punctuation, or plain English
            continue
        # Both backends strip, which would weld words together across a group
        # boundary, so the surrounding space is put back by hand.
        lead = group[: len(group) - len(group.lstrip())]
        trail = group[len(group.rstrip()) :]
        core = group.strip()
        spoken = (
            phonemize(core, language="he", model=model, normalize=normalize)
            if kind == "hebrew"
            else phonemize(core, language=language)
        )
        pieces.append(lead + spoken + trail)
    return "".join(pieces)

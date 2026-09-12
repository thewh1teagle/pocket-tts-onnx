"""Locating a model file: a local path, or the Hub copy fetched on first use.

The exported models live on the Hub at `thewh1teagle/pocket-tts-onnx`, one file
each, so a script can name the one it wants and let the download happen:

    tts = PocketTTS.from_pretrained("english")      # pocket-tts-english.onnx
    tts = PocketTTS.from_pretrained("english-ipa")  # with the Hebrew adapter

huggingface_hub caches what it fetches, so only the first run needs a network;
set HF_HOME to move the cache.
"""

from __future__ import annotations

from pathlib import Path

REPO = "thewh1teagle/pocket-tts-onnx"


def model_filename(name: str) -> str:
    """`english` -> `pocket-tts-english.onnx`; a full filename passes through."""
    return name if name.endswith(".onnx") else f"pocket-tts-{name}.onnx"


def download(name: str = "english", repo: str = REPO) -> Path:
    """The local path of a model from the Hub, downloading it on first use.

    `name` is a short model name such as `english` or `english-ipa`, or the
    exact filename in the repository.
    """
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, model_filename(name)))

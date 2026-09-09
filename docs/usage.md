# Using it

## Voices

A model file ships with voices built in; `tts.voices()` lists them.

```python
samples, sample_rate = tts.create("Hello world.", voice="alba")
```

Cloning is its own call, so synthesis never touches the encoder. That session
is built the first time you clone, and generation runs on the step graph alone.

```python
cond = tts.clone_voice("my_voice.wav")   # [1, T, model_dim]
np.save("my_voice.npy", cond)            # do this once

tts.create("Hello.", voice=np.load("my_voice.npy"))
```

`clone_voice` also takes raw samples (`clone_voice(samples, sample_rate=44100)`),
downmixes to mono and resamples to 24 kHz. It uses `scipy.signal.resample_poly`
when scipy is installed, matching upstream exactly, and a Kaiser-windowed sinc in
pure numpy otherwise. Conditioning is a plain array, so `cond[:, :1 + frames]`
shortens a prompt.

`voice=None` prompts with nothing at all, which is what an adapter trained
without voice prompts expects.

## Adapters

A file exported with `--lora <checkpoint.pt>` speaks both ways:

```python
tts.create("Ordinary English spelling.", voice=cond)                  # base model
tts.create("ʃalˈom, mˈa ʃlomχˈa hajˈom?", voice=cond, phonemes=True)  # adapter
```

`phonemes=True` turns the adapter on and switches to its tokenizer, where each
IPA, nikud and Hebrew character gets an id of its own and everything else still
goes through SentencePiece. `lora=` overrides that gate if you ever want the two
apart.

Adapter mode carries its own defaults from the adapter's inference path: no
capitalisation or trailing period, both of which corrupt phonemes; a fixed
2-frame tail; and a larger chunk budget, since one token per character adds up
faster than one per word piece.

## Getting text into phonemes

`phonemize_mixed` takes ordinary text and sends each part the shortest way it
can:

```python
from pocket_tts_onnx import phonemize_mixed

text = phonemize_mixed("אני עובד עם Photoshop כל יום.", model="renikud.onnx")
tts.create(text, voice=cond, phonemes=True)
```

| written as | becomes |
| --- | --- |
| `[[ʃalˈom]]`, `[[הַיָּם]]` | itself, brackets removed: IPA or nikud, whatever is inside is spoken as written |
| Hebrew with nikud, plain or enhanced | itself, kept exactly as typed |
| unvocalized Hebrew | phonemes from [renikud](https://huggingface.co/thewh1teagle/renikud) |
| Latin script | phonemes from espeak (pass `language=None` to leave it as written) |

Digits go first. renikud reads letters, so `₪25` would reach it as three
characters it has no consonant for and come back out as literal `₪25` for the
tokenizer to make what it can of. Hebrew text is therefore normalized before any
of that: numbers, money, dates, times and units become the words a person would
say, while `[[literals]]`, nikud, the phonikud `|` prefix, Latin runs, URLs and
emails are left exactly as written.

```python
phonemize_mixed("הכרטיס עלה ₪25 בשעה 14:30", model="renikud.onnx")
# ... עשרים וחמישה שקלים בשעה שתיים וחצי אחר הצהריים, as phonemes
```

Pass `normalize=False` to send the text through as written, or a
[`Config`](https://github.com/thewh1teagle/heb-tts-normalizer) to set the
reading style — `Config(clock=Clock.H24)` reads `14:30` as ארבע עשרה שלושים.
`normalize_hebrew` is that step on its own.

Nikud is already unambiguous, which is why it is left alone; unvocalized Hebrew
is not, which is why it needs a phonemizer. To add nikud to a line in the first
place, plain or with the phonikud stress and prefix marks, use
[phonikud](https://pypi.org/project/phonikud-onnx/). renikud's weights are a separate
download: pass `model=`, set `$RENIKUD_MODEL`, or let `huggingface_hub` fetch
them.

`phonemize` is the single-language version underneath it, if you want one script
only:

```python
phonemize("How are you today?")        # espeak
phonemize("שלום עולם", language="he")  # renikud
```

Backends are built once and reused: the first English call spends about a second
loading espeak and later ones are microseconds; renikud is 61 ms then 3 ms.

## Decode steps

The flow sampler's step count is a runtime argument:

```python
tts.create(text, voice=cond, decode_steps=2)
```

ONNX has no runtime-length loop here, so export unrolls the head
`--max-decode-steps` times and gates the copies past `decode_steps` to zero. Any
count up to the maximum is exactly equal to unrolling that count alone. But the
maximum is what you pay for, whichever count you ask for:

| exported max | steps=1 | steps=2 | steps=4 |
| --- | --- | --- | --- |
| 1 | 10.8x | | |
| 2 | 10.2x | 10.1x | |
| 4 | 9.0x | 9.2x | 9.1x |

Real-time factor, 2 threads, same sentence.

## Speed

Apple M-series, 2 threads, int8:

| | |
| --- | --- |
| time to first audio, voice already prompted | 20 ms |
| including the voice prefill | 48 ms |
| real-time factor | ~10x |
| session load | 230 ms |

`PocketTTS(path, num_threads=…)` sets the thread count; more threads buy a
little, but upstream's own figures assume two cores.

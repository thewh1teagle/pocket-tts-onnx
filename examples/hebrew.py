"""Generate Hebrew speech, in every input format the adapter understands.

`pocket-tts-english-ipa.onnx` has the Hebrew adapter and a Hebrew voice bundled
in; it is fetched from the Hub on first run and cached (or see docs/EXPORT.md to
build one and pass its path to `PocketTTS` instead). Unvocalized Hebrew is
phonemized by conikud, which fetches its own weights the same way. Then:

    uv run python examples/hebrew.py
"""

import soundfile as sf

from pocket_tts_onnx import PocketTTS, phonemize_mixed

MODEL = "english-ipa"

# A Hebrew voice, so the accent has somewhere to come from. To use your own
# recording instead, clone it once and pass the result as `voice`:
#
#   voice = tts.clone_voice("my_voice.wav")
VOICE = "omer"

# `phonemize_mixed` sends each part of the text the shortest way to phonemes it
# can, so all of these are ordinary input.
LINES = {
    # Unvocalized Hebrew: conikud guesses the vowels.
    "plain": "הכוח לשנות מתחיל ברגע שבו אתה מאמין שזה אפשרי!",
    # Latin words go through espeak, so they are spoken rather than spelled.
    "brands": "אני עובד עם Photoshop ועם Instagram כל יום.",
    # Numbers, money, dates and times are spoken as words (normalize=True is
    # the default): ₪25 becomes עשרים וחמישה שקלים, 14:30 שתיים וחצי אחר הצהריים.
    "numbers": "הכרטיס עלה ₪25 והסרט מתחיל בשעה 14:30 ב-3 באוגוסט.",
    # Nikud is already unambiguous, so it is kept exactly as typed. Both of
    # these were produced by phonikud (https://pypi.org/project/phonikud-onnx).
    "nikud": "הַיָּם הָיָה שָׁקֵט, וְהַשֶּׁמֶשׁ שָׁקְעָה מֵאֲחוֹרֵי הֶהָרִים.",
    # Enhanced nikud adds the phonikud marks on top: a prefix boundary, an ole
    # for stress, and a meteg marking a vocal shva.
    "nikud_enhanced": "סֵ֫פֶר טוֹב יָכוֹל לְֽשַׁנּוֹת אֶת הַ|דֶּ֫רֶךְ שֶׁ|בָּהּ אַתָּה חוֹשֵׁב עַל הָ|עוֹלָם.",
    # Double brackets hold IPA, for when you want to fix one word yourself.
    "literal": "המילה [[ʃalˈom]] נשמעת ככה.",
}

TEMPERATURE = 0.3
DECODE_STEPS = 2


def main() -> None:
    tts = PocketTTS.from_pretrained(MODEL)

    for name, line in LINES.items():
        phonemes = phonemize_mixed(line)
        print(f"{line}\n  -> {phonemes}")
        samples, sample_rate = tts.create(
            phonemes,
            voice=VOICE,
            phonemes=True,
            temperature=TEMPERATURE,
            decode_steps=DECODE_STEPS,
        )
        sf.write(f"{name}.wav", samples, sample_rate)
        print(f"  wrote {name}.wav, {len(samples) / sample_rate:.2f}s")


if __name__ == "__main__":
    main()

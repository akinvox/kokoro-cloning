"""Normalize punctuation to the Kokoro vocabulary."""

ALIASES = {"[": "(", "]": ")"}


def canonicalize_punctuation(phonemes):
    return phonemes.translate(str.maketrans(ALIASES))


def phonemize_native(g2p, text):
    phonemes, _ = g2p(text)
    return canonicalize_punctuation(phonemes)

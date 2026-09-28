"""Exact retained native punctuation ABI; no speech-phone aliases."""
CONTRACT = 'native-kokoro-square-bracket-parentheses-v1'
ALIASES = {'[': '(', ']': ')'}

def canonicalize_punctuation(phonemes):
    return phonemes.translate(str.maketrans(ALIASES))

def phonemize_native(g2p, text):
    phonemes, _ = g2p(text)
    return canonicalize_punctuation(phonemes)

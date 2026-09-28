# Listening comparisons

Every example has three players: **reference**, **clone · same words**, and
**clone · new words**. All audio is synthetic. Clones use the released 100k
model with seed `20260926` and normal reference enrollment.

| Example | Reference | Same words | New words |
|---|---|---|---|
| Clean neutral | [Reference](../examples/reference.wav) | [Clone](neutral_same_text.wav) | [Clone](neutral_clone.wav) |
| Story narration | [Reference](warm_reference.wav) | [Clone](warm_same_text.wav) | [Clone](story_clone.wav) |
| Dialogue | [Reference](warm_reference.wav) | [Clone](warm_same_text.wav) | [Clone](dialogue_clone.wav) |
| Longer passage | [Reference](warm_reference.wav) | [Clone](warm_same_text.wav) | [Clone](longer_clone.wav) |
| Explanation | [Reference](clear_reference.wav) | [Clone](clear_same_text.wav) | [Clone](explanation_clone.wav) |
| Questions | [Reference](clear_reference.wav) | [Clone](clear_same_text.wav) | [Clone](questions_clone.wav) |
| Expression | [Reference](expressive_reference_v2.wav) | [Clone](expressive_same_text_v2.wav) | [Clone](expressive_new_text_v2.wav) |
| Whispering | [Reference](whisper_reference_v2.wav) | [Clone](whisper_clone_v2.wav) | [Clone](whisper_new_text.wav) |
| Accent | [Reference](australian_reference.wav) | [Clone](accent_clone.wav) | [Clone](accent_new_text.wav) |

The same-word player uses the reference transcript. It helps reveal differences
in pronunciation, timing and delivery. The new-word player shows how the voice
carries onto another passage. Examples sharing a reference also share its
same-word clone.

Compare voice character and clarity across the narration examples. Expression,
whispering and accent examples show limitations: pauses and emphasis can drift,
whispers can become voiced speech, and the original accent may not carry over.
These examples are not a broad speaker benchmark.

The neutral reference uses original Kokoro `af_heart`. Other references were
made with [Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign).
Transcripts are included beside the WAVs; `expanded_samples.json` records source
provenance and target texts, and `sample_manifest.json` lists audio checksums.

After following the main installation guide, generate both versions:

```bash
kokoro-clone --reference samples/warm_reference.wav --reference-text-file samples/warm_reference.txt --text-file samples/warm_reference.txt --output same_words.wav
kokoro-clone --reference samples/warm_reference.wav --reference-text-file samples/warm_reference.txt --text-file samples/story_text.txt --output new_words.wav
```

Waveforms can differ across hardware and software versions.

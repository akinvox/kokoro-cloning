# Listening comparisons

Each example has its own synthetic voice and passage. Play the reference,
the same-sentence clone, then the different-sentence clone.

| Example | Reference | Same sentence | Different sentence |
|---|---|---|---|
| Warm narration | [Reference](story_reference.wav) | [Clone](story_same.wav) | [Clone](story_new.wav) |
| Lively dialogue | [Reference](dialogue_reference.wav) | [Clone](dialogue_same.wav) | [Clone](dialogue_new.wav) |
| A longer passage | [Reference](longer_reference.wav) | [Clone](longer_same.wav) | [Clone](longer_new.wav) |
| Factual explanation | [Reference](explanation_reference.wav) | [Clone](explanation_same.wav) | [Clone](explanation_new.wav) |
| Questions and answers | [Reference](questions_reference.wav) | [Clone](questions_same.wav) | [Clone](questions_new.wav) |
| Quick-start example | [Reference](../examples/reference.wav) | [Clone](neutral_same.wav) | [Clone](neutral_new.wav) |
| Expressive delivery | [Reference](expression_reference.wav) | [Clone](expression_same.wav) | [Clone](expression_new.wav) |
| Whisper preservation | [Reference](whisper_reference.wav) | [Clone](whisper_same.wav) | [Clone](whisper_new.wav) |
| Accent preservation | [Reference](accent_reference.wav) | [Clone](accent_same.wav) | [Clone](accent_new.wav) |

The strength samples are selected examples. Similarity varies with the reference.
The weakness examples show drift in expression, whispering and accents.

All references are AI-generated: the quick-start voice uses stock Kokoro
`af_heart`; the other voices use [Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign). `sample_manifest.json`
records their provenance, transcripts, selected seeds and audio checksums.

After following the main installation guide, reproduce a pair:

```bash
kokoro-clone --reference samples/dialogue_reference.wav --reference-text-file samples/dialogue_reference.txt --text-file samples/dialogue_reference.txt --output same_sentence.wav
kokoro-clone --reference samples/dialogue_reference.wav --reference-text-file samples/dialogue_reference.txt --text-file samples/dialogue_new.txt --seed 20260927 --output different_sentence.wav
```

Use the seed recorded in the manifest for each selected new-text sample.
Waveforms can differ across hardware and software versions.

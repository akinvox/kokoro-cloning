# Quick listening comparison

All audio is synthetic. Clones use the v1 model with seed `20260926`.
The players compare each reference with the generated speech; transcripts
and file checksums are included alongside the audio.

| Example | Reference | Clone | What to compare |
|---|---|---|---|
| Clean neutral / new words | [Kokoro reference](../examples/reference.wav) | [New-sentence clone](neutral_clone.wav) | Voice character and clear new words; also a reproducible installation example |
| Slow expressive / same words | [Expressive reference](expressive_reference_v2.wav) | [Same-text clone](expressive_same_text_v2.wav) | Pace, pauses, stretched words, breathiness and emphasis |
| Slow expressive / new words | [Same expressive reference](expressive_reference_v2.wav) | [New-sentence clone](expressive_new_text_v2.wav) | Identity and delivery when the words change |

The neutral reference was generated with original Kokoro v1.0 `af_heart`.
The expressive reference was generated with
[Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
and is 10.96 seconds long. Neither reference represents a recording of a real
person. These few examples are not a speaker benchmark or a guarantee of
unseen-speaker performance. Highly expressive delivery remains a known
limitation; listen to the differences rather than treating intelligibility
as proof of an exact match.

After following the main installation guide, reproduce the expressive examples:

```bash
kokoro-clone --reference samples/expressive_reference_v2.wav --reference-text-file samples/expressive_reference_v2.txt --text-file samples/expressive_reference_v2.txt --output expressive_same_text_v2.wav
kokoro-clone --reference samples/expressive_reference_v2.wav --reference-text-file samples/expressive_reference_v2.txt --text-file samples/expressive_new_text_v2.txt --output expressive_new_text_v2.wav
```

The neutral command is in the main guide. Waveforms may differ across execution
environments. `sample_manifest.json` records the distributed files and hashes.

## Five additional new-text examples

Five new-text examples using two synthetic Qwen voices.

| Example | Reference | Clone | Listen for |
|---|---|---|---|
| Story narration | [Reference](warm_reference.wav) | [Clone](story_clone.wav) | Listen for a consistent voice through descriptive narration. |
| Quoted dialogue | [Reference](warm_reference.wav) | [Clone](dialogue_clone.wav) | Listen for readable phrasing around quoted speech. This is one voice reading both characters. |
| Several sentences | [Reference](warm_reference.wav) | [Clone](longer_clone.wav) | Listen for voice consistency across a longer three-sentence passage. |
| Clear explanation | [Reference](clear_reference.wav) | [Clone](explanation_clone.wav) | Compare the lower voice character while it explains something in new words. |
| Questions and answers | [Reference](clear_reference.wav) | [Clone](questions_clone.wav) | Listen for clear words and how the voice handles questions and sentence boundaries. |

## Additional weaknesses

- **Whisper preservation:** [reference](whisper_reference_v2.wav) / [same-text clone](whisper_clone_v2.wav). Compare breathiness and voicing: v1 can turn a whispered reference into more normally voiced speech.
- **Accent preservation:** [reference](australian_reference.wav) / [same-text clone](accent_clone.wav). Compare the vowels in car, water, work and boat. This release uses a US-English phonemizer; preserving a reference accent is not guaranteed.

Reference transcripts are supplied alongside each WAV. `expanded_samples.json` records the target texts, original synthetic-reference provenance, release update and audio hashes.

# Third-party notices

The Apache-2.0 license covers AkinVox's original contributions and released
adapter/reference components. It does not relicense third-party dependencies.

| Component | Use | License / source |
|---|---|---|
| Kokoro | Original backbone and stock inference | [Apache-2.0](https://github.com/hexgrad/kokoro/blob/main/LICENSE); [model card](https://huggingface.co/hexgrad/Kokoro-82M) |
| StyleTTS 2, including its iSTFTNet integration | Adapted native inference and acoustic encoder implementation | [MIT, Copyright 2023 Aaron (Yinghao) Li](licenses/STYLE_TTS2_LICENSE) |
| WavLM speaker verification | Separately downloaded reference identity features | [Microsoft MIT license linked by the model card](https://huggingface.co/microsoft/wavlm-base-plus-sv#license) |
| Misaki | English phonemization | [Apache-2.0](https://github.com/hexgrad/misaki) |
| Silero VAD | Reference speech-duration check | [MIT](https://github.com/snakers4/silero-vad) |
| eSpeak NG / phonemizer | Fallback pronunciation, installed through Misaki dependencies | [eSpeak NG GPL-3.0](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING), [phonemizer GPL-3.0](https://github.com/bootphon/phonemizer/blob/master/LICENSE) |

PyTorch, Transformers, spaCy, NumPy, SciPy, SoundFile and the other installed
packages retain their respective upstream licenses. Review those terms when
redistributing a complete environment or application.

No ASR weights are included. CrisperWhisper is an optional external transcription
tool; its code and particular model checkpoints have separate licenses. In
particular, do not assume every CrisperWhisper checkpoint permits commercial
use. A checked manual transcript works without any ASR dependency.

The example reference is synthetic audio generated with original Kokoro v1.0's
`af_heart` voice. The transcript and example target text were written for this
release. It is an installation example, not evidence of real-speaker cloning
quality. No private reference recordings or training corpus are distributed.

The expressive demo reference is synthetic audio generated with [Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), whose model card declares Apache-2.0. No Qwen model weights are required or bundled.

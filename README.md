# AkinVox Kokoro Cloning v1

Clone an English reference voice onto new text with a switchable Kokoro adapter.
The release includes the **100k production adapter**, reference mapper and reference encoders, plus the inference code needed to use them. It downloads the exact original Kokoro base automatically.

[Download weights](https://huggingface.co/AKinvox/kokoro-cloning-v1) · [AkinVox](https://akinvox.com)

## Listen first

[Open the side-by-side listening page](https://akinvox.github.io/kokoro-cloning/) or use the files below. All audio, including the references, is **AI-generated**.

The samples cover five distinct reference voices and passages: warm narration,
lively dialogue, a sustained longer passage, factual explanation, and questions.
Each includes the reference, a same-sentence clone and a different-sentence clone.
The page also includes the quick-start voice and separate expression, whisper
and accent limitations. No audio file is repeated across examples.


V1 supports reference-to-new-text speech, reusable enrollment and an unchanged stock mode. Expressive fidelity and consistency across speakers remain limitations. These examples let you hear both the workflow and its limits; they are not a broad quality benchmark. [Transcripts, provenance and reproduction](samples/README.md).

## Quick start

Use Python 3.10–3.12 on Linux or WSL. CPU works; an NVIDIA GPU is optional. Allow roughly 5 GB for a CPU installation and model downloads, and at least 8 GB of system RAM. The first run downloads models; later runs reuse the cache.

### 1. Install

```bash
git clone https://github.com/akinvox/kokoro-cloning.git
cd kokoro-cloning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-tested.txt .
python -m spacy download en_core_web_sm
```

For an NVIDIA GPU, install the matching [PyTorch 2.6.0 CUDA build](https://pytorch.org/get-started/previous-versions/#v260) instead of the CPU command, then add `--device cuda` below. This guide's tested baseline is CPU on Linux/WSL.

### 2. Try the included example

```bash
kokoro-clone --reference examples/reference.wav --reference-text-file examples/reference.txt --text-file examples/text.txt --output clone.wav
```

Open `clone.wav`. The example reference is **AI-generated stock Kokoro speech**, supplied to make the first run reproducible; it is not a recording of a real speaker or a benchmark of unseen-speaker quality.

### 3. Use your own voice

Record one speaker for **5–15 seconds** in a quiet room. Use a WAV file with at least 3 seconds of actual speech and at most 30 seconds total. Save the exact words spoken in `reference.txt`, then run:

```bash
kokoro-clone --reference reference.wav --reference-text-file reference.txt --text "This is a new sentence in the reference voice." --output clone.wav
```

The transcript describes the **reference recording**; `--text` is what you want generated. Keep audible “um” and “uh” in the transcript. If you use CrisperWhisper to draft it, check its output against the final recording and review that model's separate license. ASR is optional and is not installed or bundled here.

### Original Kokoro: adapter off

```bash
kokoro-clone --adapter off --voice af_heart --text "This uses the original Kokoro voice." --output stock.wav
```

Adapter on is the default for cloning. Adapter off uses the original Kokoro runtime and English stock voicepacks, without reference conditioning. Both modes read the same pinned base file; nothing is merged into it. Use this package's loader: the adapter also needs its reference mapper/encoders and cannot be dropped directly into an arbitrary Kokoro or PEFT loader.

## Python

```python
from kokoro_cloning import KokoroCloner

model = KokoroCloner()  # CPU; use device="cuda" for a compatible GPU
reference = model.enroll("examples/reference.wav", open("examples/reference.txt").read())
model.save("A different sentence, spoken with the reference voice.", "clone.wav", reference=reference)

model.set_adapter_enabled(False)
model.save("Original Kokoro again.", "stock.wav", voice="af_heart")

model.set_adapter_enabled(True)
model.save("Back to the cloned voice.", "clone.wav", reference=reference)
```

The reference encoders and WavLM are needed during enrollment to extract voice information. The mapper converts those observations into reusable conditioning; the adapter alone cannot clone a recording. Once enrolled, generation uses that conditioning without re-encoding the audio.

Enroll once and reuse the returned conditioning for multiple sentences. Each instance serializes generation and mode switching. Stock mode loads an additional original-runtime graph on first use, so using both modes takes more RAM than cloning alone.

## Local browser demo

```bash
python -m pip install -c constraints-tested.txt '.[demo]'
kokoro-clone-demo
```

Open the localhost address printed in the terminal. Upload a reference, enter its transcript and the new text, then generate. Reference processing happens on the machine running the demo. The default demo binds to localhost.

## What is included

| File | Purpose |
|---|---|
| `adapter.pt` | Switchable low-rank updates for native Kokoro inference |
| `reference_mapper.pt` | Maps reference observations into voice conditioning and reference memory |
| `reference_encoders.pt` | Required frozen acoustic reference encoders |
| `config.json` | Architecture, adapter targets, pinned dependencies and checksums |

The three weight files total about **202 MB**. The runtime separately downloads pinned [Kokoro v1.0](https://huggingface.co/hexgrad/Kokoro-82M) and [WavLM speaker features](https://huggingface.co/microsoft/wavlm-base-plus-sv). There are no merged backbone weights, training scripts, optimizer states or training datasets in this release.

We adapted Kokoro using LoRA while keeping its original backbone tensors frozen. A learned reference mapper provides speaker style and acoustic/text reference memory to the native synthesis path. This keeps adaptation separate from the base and supports future adapter development. Arbitrary adapter combinations, different Kokoro bases and other languages are not validated by this release.

## Limitations

- V1 supports **English voice cloning**.
- Strong emotion, shouting, whispering, singing and highly expressive references can be difficult to follow. Identity, accent and delivery may drift.
- Noise, music, several speakers, very short references and inaccurate transcripts reduce quality. The speech detector may reject weak or unusual speech; try a longer clean recording.
- Names, unusual words and punctuation may need spelling adjustments. Split long passages into sentences; each request must fit the model's 510-token limit.
- Output is 24 kHz mono. Long-form consistency and reliable transfer to every unseen speaker are not guaranteed. Listen to generated audio before using it.
- Results can differ across hardware and software versions. Exact stock comparisons use the same seed, voice, text and runtime environment.

Version 2 is planned with substantial improvements. Training code is not included while the research is still being consolidated; I intend to release it when it is ready.

## About this project

I’m building this for my audiobook site, [akinvox.com](https://akinvox.com). The goal is voice cloning without sacrificing quality or speed. V1 has had 100k cloning updates on pretrained Kokoro; I believe more training could improve it substantially. Work on v2 is ongoing.

The site is still under construction, and I’d appreciate you giving it a go.

## License and credits

AkinVox's code and released weights are **Apache-2.0**, including commercial use. Upstream components retain their own licenses and notices; see [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_LICENSES.md). Use reference recordings you have permission to use, and identify synthetic audio appropriately.

Built on Kokoro, StyleTTS 2, iSTFTNet, WavLM and Misaki. [Report an issue](https://github.com/akinvox/kokoro-cloning/issues) with your environment and a reproducible example; do not post private recordings without permission.

# Release validation

Cloning v1 uses production update **100000**, checkpoint SHA256
`8841a672700b42eb89db94c8ac1e2f7defd41f8b791bdd5f8f9e93f63b727d73`.
It is the completed production run, not a later experimental research model.

| Check | Result |
|---|---|
| Original stock tensors | All 548 native backbone tensors unchanged in the source checkpoint |
| Extracted components | 199 adapter targets; reference mapper and two required acoustic encoders; strict state loading and SHA256 verification |
| Full checkpoint versus extracted release | Both acoustic encoder outputs, mapped style and reference memory matched exactly; duration, F0, excitation and waveform matched on two synthesis texts in the same environment |
| Original stock mode | Waveform and predicted duration matched upstream Kokoro 0.9.4 exactly in 12 English voice/text cases |
| Mode switching | Stock restored exactly after toggling; cloning output unchanged through on/off/on |
| Clean installation | Isolated Python 3.10 environment, PyTorch 2.6 CPU, fresh model cache; both documented CLI examples completed |
| Browser demo | Reference cloning and stock-mode requests completed; test server exited afterward |
| Listening page | 27 players covering 21 unique audio files loaded in Chromium; desktop and 390px mobile layouts checked |

Comparisons use matched text, conditioning, seed and execution environment.
Full-state/export parity was checked together on PyTorch 2.12 CPU; the public
guide and deployment checks use the pinned PyTorch 2.6 CPU environment.
Exact samples are not promised across different versions or hardware.

Stock mode deliberately uses the original upstream inference implementation.
The inherited cloning implementation can produce small waveform differences
even with identical stock tensors, so a weight-only check was not treated as
sufficient to establish upstream stock-output preservation.

These are integrity, reproducibility and usability checks. They are not a
universal identity, expressiveness or naturalness score. The listening samples
are illustrative synthetic references, and the model card states v1's limits.

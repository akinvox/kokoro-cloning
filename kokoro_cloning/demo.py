"""Optional local Gradio interface."""

import argparse
import threading
from functools import lru_cache

import torch

from .api import KokoroCloner


def build_demo(*, device="cpu", model_dir=None, base_dir=None, wavlm_dir=None):
    import gradio as gr

    lock = threading.Lock()

    @lru_cache(maxsize=1)
    def load():
        return KokoroCloner(
            model_dir, base_dir=base_dir, wavlm_dir=wavlm_dir, device=device
        )

    def generate(mode, audio, transcript, text, voice):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise gr.Error(
                "Enter a short sentence or paragraph (up to 2,000 characters)."
            )
        if mode == "Clone reference" and (
            not audio or not isinstance(transcript, str) or not transcript.strip()
        ):
            raise gr.Error("Upload a reference and enter its exact transcript.")
        try:
            with lock:
                model = load()
                model.set_adapter_enabled(mode == "Clone reference")
                if model.adapter_enabled:
                    reference = model.enroll(audio, transcript)
                    wave = model.generate(text, reference=reference)
                else:
                    wave = model.generate(text, voice=voice)
                return (model.sample_rate, wave)
        except (ValueError, OSError, RuntimeError) as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(
        title="AkinVox Kokoro Cloning v1", delete_cache=(3600, 3600)
    ) as demo:
        gr.Markdown(
            "# AkinVox Kokoro Cloning v1\nClone an English reference onto new words, or use an original Kokoro voice. "
            "Built for [AkinVox](https://akinvox.com), an audiobook site under construction."
        )
        mode = gr.Radio(
            ["Clone reference", "Original Kokoro (adapter off)"],
            value="Clone reference",
            label="Mode",
        )
        audio = gr.Audio(
            type="filepath", label="Reference: one speaker, 5–15 seconds recommended"
        )
        transcript = gr.Textbox(label="Exact words spoken in the reference", lines=3)
        text = gr.Textbox(
            label="New text to speak",
            value="Welcome to AkinVox. Every story opens a new world.",
            lines=3,
        )
        voice = gr.Dropdown(
            ["af_heart", "af_bella", "am_adam", "am_michael", "bf_alice", "bm_george"],
            value="af_heart",
            label="Stock voice (used only with adapter off)",
        )
        button = gr.Button("Generate speech", variant="primary")
        output = gr.Audio(label="AI-generated speech", type="numpy")
        button.click(
            generate,
            [mode, audio, transcript, text, voice],
            output,
            concurrency_limit=1,
            api_name="generate",
        )
        gr.Markdown(
            "Use recordings you have permission to use. Highly expressive references may be difficult to follow. "
            "Processing happens on the machine hosting this demo. Uploaded and generated files may be cached there for up to an hour. "
            "[Guide and source](https://github.com/akinvox/kokoro-cloning) · "
            "[Weights](https://huggingface.co/AKinvox/kokoro-cloning-v1)"
        )
    return demo.queue(max_size=8, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description="Run the AkinVox cloning demo locally")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    torch.set_num_threads(2)
    build_demo(device=args.device).launch(
        server_name="127.0.0.1", server_port=args.port
    )


if __name__ == "__main__":
    main()

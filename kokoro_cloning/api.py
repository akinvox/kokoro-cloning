"""Reference cloning and original stock TTS on one immutable native backbone."""

import gc
import hashlib
import io
import random
import re
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import torchaudio
from munch import munchify
from transformers import AlbertConfig, AlbertModel

from . import adapters, assets, inference_helpers
from .en_models import build_model
from .en_predict import predict_speech
from .frontend_punctuation import phonemize_native
from .kokoro_symbols import TextCleaner, unknown_symbols
from .reference_mel import preprocess
from .reference_pitch_text import attach_reference_text, encode_reference
from .reference_sequence import SequenceReferenceMapper

ROOTS = ("bert", "bert_encoder", "predictor", "decoder", "text_encoder")


class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).last_hidden_state


def seed(value):
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(value)


def stock_keys(state):
    result = {}
    for name, value in state.items():
        name = name.removeprefix("module.")
        if name.endswith("weight_g"):
            name = name[:-8] + "parametrizations.weight.original0"
        elif name.endswith("weight_v"):
            name = name[:-8] + "parametrizations.weight.original1"
        result[name] = value
    return result


class KokoroCloner:
    """Use ``enroll`` once, then ``generate``; select stock mode explicitly.

    English cloning is qualified. Stock mode supports the bundled English
    frontend and original English voicepacks. Calls on an instance are serialized
    so adapter switching cannot mix requests. Output is mono float32 at 24 kHz.
    """

    sample_rate = 24000

    def __init__(
        self,
        model_dir=None,
        *,
        base_dir=None,
        wavlm_dir=None,
        device="cpu",
        local_files_only=False,
    ):
        self.device = str(torch.device(device))
        self.base_dir, self.wavlm_dir = base_dir, wavlm_dir
        self.local_files_only = local_files_only
        self._lock = threading.RLock()
        self.config, files = assets.release_assets(
            model_dir, local_files_only=local_files_only
        )
        base = assets.base_file(
            "kokoro-v1_0.pth",
            base_dir,
            expected=assets.BASE_SHA256,
            local_files_only=local_files_only,
        )
        self._base_path, self._stock_model = base, None
        p = self.config["architecture"]
        self.core = build_model(
            munchify(p),
            CustomAlbert(AlbertConfig(vocab_size=p["n_token"], **p["plbert"])),
        )
        state = torch.load(base, map_location="cpu", weights_only=True, mmap=True)
        if set(state) != set(ROOTS):
            raise ValueError("Unexpected stock backbone roots")
        for name in ROOTS:
            self.core[name].load_state_dict(stock_keys(state[name]), strict=True)
        del state
        delta = torch.load(
            files["adapter.pt"], map_location="cpu", weights_only=True, mmap=True
        )
        self.adapters = adapters.attach(self.core, self.config["targets"], delta)
        observations = torch.load(
            files["reference_encoders.pt"],
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        for name in ("style_encoder", "predictor_encoder"):
            self.core[name].load_state_dict(observations[name], strict=True)
        mapper = torch.load(
            files["reference_mapper.pt"],
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        self.mapper = SequenceReferenceMapper(mapper["pooled.base_table"].clone())
        attach_reference_text(self.mapper)
        self.mapper.load_state_dict(mapper, strict=True)
        for module in [*self.core.values(), self.mapper]:
            module.eval().requires_grad_(False).to(self.device)
        self.cleaner, self._frontends, self._packs = TextCleaner(), {}, {}
        self.adapter_enabled = True

    def set_adapter_enabled(self, enabled):
        if type(enabled) is not bool:
            raise TypeError("enabled must be a bool")
        with self._lock:
            for adapter in self.adapters:
                adapter.enabled = enabled
            self.adapter_enabled = enabled
        return self

    def phonemize(self, text, *, british=False):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a nonempty string")
        if british not in self._frontends:
            from misaki.en import G2P
            from misaki.espeak import EspeakFallback

            self._frontends[british] = G2P(
                trf=False,
                british=british,
                fallback=EspeakFallback(british=british),
                unk="",
            )
        phones = phonemize_native(self._frontends[british], text.strip())
        self.common(phones)
        return phones

    def common(self, phones):
        if not isinstance(phones, str) or not phones or unknown_symbols(phones):
            raise ValueError("Unsupported or empty phonemes")
        ids = [0, *self.cleaner(phones), 0]
        if not 3 <= len(ids) <= 510:
            raise ValueError(
                "Text must fit 3–510 tokens; split it into shorter sentences"
            )
        text = torch.tensor([ids], dtype=torch.long, device=self.device)
        return dict(
            text=text,
            lengths=torch.tensor([len(ids)], device=self.device),
            text_mask=torch.zeros_like(text, dtype=torch.bool),
        )

    @torch.inference_mode()
    def enroll_features(self, wave24, reference_phonemes, features):
        """Create voice conditioning from audio, phonemes and speaker features."""
        with self._lock:
            wave24 = np.asarray(wave24, dtype=np.float32)
            if (
                wave24.ndim != 1
                or not np.isfinite(wave24).all()
                or not 3 <= len(wave24) / 24000 <= 30
            ):
                raise ValueError(
                    "Reference must be finite mono speech lasting 3–30 seconds"
                )
            rc = self.common(reference_phonemes)
            mel = preprocess(wave24).to(self.device)
            prompt = encode_reference(
                self.mapper,
                mel,
                torch.tensor([mel.shape[-1]], device=self.device),
                rc["text"],
                rc["text_mask"],
            )
            values = [
                features[k].to(self.device) for k in ("wavlm", "raw_sdec", "raw_spred")
            ]
            style = self.mapper(*values, rc["lengths"] - 1, prompt)
            if style.shape != (1, 256) or not torch.isfinite(style).all():
                raise RuntimeError("Invalid reference conditioning")
            return dict(
                format="akinvox-cloning-reference-v1",
                adapter_sha256=self.config["files"]["adapter.pt"]["sha256"],
                mapper_sha256=self.config["files"]["reference_mapper.pt"]["sha256"],
                style=style.cpu(),
                prompt=tuple(v.cpu() for v in prompt),
                reference_phonemes=reference_phonemes,
            )

    @torch.inference_mode()
    def enroll(self, reference_audio, reference_text):
        """Enroll final reference audio with its checked, verbatim transcript."""
        with self._lock:
            phones = self.phonemize(reference_text)
            info = sf.info(reference_audio)
            if info.channels > 8 or not 3 <= info.duration <= 30:
                raise ValueError("Use one speaker in a 3–30 second recording")
            wave, rate = sf.read(reference_audio, dtype="float32", always_2d=True)
            wave = wave.mean(axis=1)
            if (
                not np.isfinite(wave).all()
                or np.sqrt(np.mean(wave**2)) < 1e-4
                or np.mean(np.abs(wave) >= 0.999) > 0.001
            ):
                raise ValueError("Reference is silent, clipped or non-finite")
            if rate != 24000:
                wave = soxr.resample(wave, rate, 24000).astype(np.float32)
            prepared = io.BytesIO()
            sf.write(prepared, wave, 24000, format="WAV", subtype="PCM_16")
            audio_hash = hashlib.sha256(prepared.getvalue()).hexdigest()
            prepared.seek(0)
            wave, _ = sf.read(prepared, dtype="float32")
            audio16 = torchaudio.functional.resample(
                torch.from_numpy(wave), 24000, 16000
            )
            from silero_vad import get_speech_timestamps, load_silero_vad

            vad = load_silero_vad().cpu().eval()
            spans = get_speech_timestamps(audio16, vad, sampling_rate=16000)
            speech = sum(s["end"] - s["start"] for s in spans) / 16000
            del vad
            if speech < 3:
                raise ValueError(
                    "Less than 3 seconds of detected speech; use a longer clean reference"
                )
            mel = preprocess(np.pad(wave, (5000, 5000))).squeeze()
            mel = mel[:, : mel.shape[-1] - mel.shape[-1] % 2][None, None].to(
                self.device
            )
            features = dict(
                raw_sdec=self.core.style_encoder(mel).float(),
                raw_spred=self.core.predictor_encoder(mel).float(),
            )
            from transformers import AutoFeatureExtractor, WavLMForXVector

            identity_dir = assets.wavlm_files(
                self.config, self.wavlm_dir, local_files_only=self.local_files_only
            )
            extractor = AutoFeatureExtractor.from_pretrained(
                identity_dir, local_files_only=True
            )
            identity = (
                WavLMForXVector.from_pretrained(
                    identity_dir, local_files_only=True, use_safetensors=False
                )
                .cpu()
                .eval()
                .requires_grad_(False)
            )
            inputs = extractor(
                audio16.numpy(), sampling_rate=16000, return_tensors="pt"
            )
            embedding = identity(
                input_values=inputs.input_values,
                attention_mask=getattr(inputs, "attention_mask", None),
            ).embeddings.float()
            features["wavlm"] = torch.nn.functional.normalize(embedding, dim=-1)
            del identity, extractor
            artifact = self.enroll_features(wave, phones, features)
            artifact.update(
                reference_text=reference_text.strip(),
                audio_sha256=audio_hash,
                reference_seconds=len(wave) / 24000,
                detected_speech_seconds=speech,
            )
            gc.collect()
            return artifact

    @torch.inference_mode()
    def synthesize_phones(self, reference, phones, *, seed_value=20260926):
        with self._lock:
            if not self.adapter_enabled:
                raise ValueError(
                    "Cloning requires adapter on; stock mode uses an original voicepack"
                )
            if (
                reference.get("format") != "akinvox-cloning-reference-v1"
                or reference.get("adapter_sha256")
                != self.config["files"]["adapter.pt"]["sha256"]
                or reference.get("mapper_sha256")
                != self.config["files"]["reference_mapper.pt"]["sha256"]
            ):
                raise ValueError(
                    "Reference conditioning belongs to a different model; enroll it again"
                )
            style = reference["style"].to(self.device)
            prompt = tuple(v.to(self.device) for v in reference["prompt"])
            if style.shape != (1, 256) or not torch.isfinite(style).all():
                raise ValueError("Invalid reference style")
            seed(seed_value)
            got = predict_speech(
                self.core,
                inference_helpers,
                self.common(phones),
                style[:, :128],
                style[:, 128:],
                self.mapper,
                prompt,
            )
            if not all(torch.isfinite(v).all() for v in got.values()):
                raise RuntimeError("Non-finite synthesis output")
            return {k: v.detach().cpu() for k, v in got.items()}

    @torch.inference_mode()
    def stock_phones(self, phones, voice="af_heart", *, seed_value=20260926):
        with self._lock:
            if self.adapter_enabled:
                raise ValueError(
                    "Set adapter off before requesting original stock output"
                )
            if not re.fullmatch(r"[ab][fm]_[a-z]+", voice):
                raise ValueError(
                    "This release supports original English stock voice IDs"
                )
            name = f"voices/{voice}.pt"
            if name not in self.config["stock_voices"]:
                raise ValueError("Unknown stock voice")
            if voice not in self._packs:
                p = assets.base_file(
                    name,
                    self.base_dir,
                    expected=self.config["stock_voices"][name]["sha256"],
                    local_files_only=self.local_files_only,
                )
                pack = torch.load(p, map_location="cpu", weights_only=True)
                if pack.shape != (510, 1, 256) or not torch.isfinite(pack).all():
                    raise ValueError("Invalid stock voicepack")
                self._packs[voice] = pack
            self.common(phones)
            if self._stock_model is None:
                from kokoro import KModel

                config_path = assets.base_file(
                    "config.json",
                    self.base_dir,
                    expected=self.config["stock_config_sha256"],
                    local_files_only=self.local_files_only,
                )
                self._stock_model = (
                    KModel(
                        repo_id=assets.BASE_REPO,
                        config=str(config_path),
                        model=str(self._base_path),
                    )
                    .to(self.device)
                    .eval()
                    .requires_grad_(False)
                )
            style = self._packs[voice][len(phones) - 1].to(self.device)
            seed(seed_value)
            output = self._stock_model(phones, style, return_output=True)
            result = dict(duration=output.pred_dur, wave=output.audio)
            if not all(torch.isfinite(v).all() for v in result.values()):
                raise RuntimeError("Non-finite stock output")
            return {k: v.detach().cpu() for k, v in result.items()}

    def generate(self, text, *, reference=None, voice="af_heart", seed_value=20260926):
        with self._lock:
            if self.adapter_enabled:
                if reference is None:
                    raise ValueError("Enroll reference audio and its transcript first")
                return self.synthesize_phones(
                    reference, self.phonemize(text), seed_value=seed_value
                )["full_wave"].numpy()
            if reference is not None:
                raise ValueError(
                    "Stock mode takes a voice ID; omit reference conditioning"
                )
            return self.stock_phones(
                self.phonemize(text, british=voice.startswith("b")),
                voice,
                seed_value=seed_value,
            )["wave"].numpy()

    def save(self, text, output, **kwargs):
        wave = self.generate(text, **kwargs)
        sf.write(output, wave, self.sample_rate, subtype="PCM_16")
        return Path(output)

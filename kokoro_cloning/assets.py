"""Pinned public assets with full SHA256 checks, including offline inputs."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODEL_REPO = "AKinvox/kokoro-cloning-v1"
MODEL_REVISION = "v1.0.0"
BASE_REPO = "hexgrad/Kokoro-82M"
BASE_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
BASE_SHA256 = "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
WAVLM_REPO = "microsoft/wavlm-base-plus-sv"
WAVLM_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, expected):
    path = Path(path)
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Checksum mismatch or missing file: {path}")
    return path


def release_assets(model_dir=None, *, local_files_only=False):
    from .release_pin import CONFIG_SHA256

    def get(name):
        if model_dir is not None:
            return Path(model_dir) / name
        return Path(
            hf_hub_download(
                MODEL_REPO,
                name,
                revision=MODEL_REVISION,
                local_files_only=local_files_only,
            )
        )

    config = json.loads(verify(get("adapter_config.json"), CONFIG_SHA256).read_text())
    if (
        config["format"] != "akinvox-kokoro-cloning-adapter-v1"
        or config["update"] != 100000
    ):
        raise ValueError("Unsupported release configuration")
    names = ("adapter.pt", "reference_mapper.pt", "reference_encoders.pt")
    files = {n: verify(get(n), config["files"][n]["sha256"]) for n in names}
    return config, files


def base_file(name, base_dir=None, *, expected, local_files_only=False):
    if base_dir is not None:
        path = Path(base_dir) / name
    else:
        path = hf_hub_download(
            BASE_REPO, name, revision=BASE_REVISION, local_files_only=local_files_only
        )
    return verify(path, expected)


def wavlm_files(config, folder=None, *, local_files_only=False):
    if folder is None:
        folder = snapshot_download(
            WAVLM_REPO,
            revision=WAVLM_REVISION,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "pytorch_model.bin",
            ],
            local_files_only=local_files_only,
        )
    folder = Path(folder)
    verify(folder / "pytorch_model.bin", config["wavlm_weights_sha256"])
    metadata_hashes = {
        "config.json": "c6ac8c0ce55b18d4612677931036fe5ce78093e87361c552a173f00a0bb07514",
        "preprocessor_config.json": "99272fe8ccfab114b68b478681ea47ee3a1ce62bb788cb92dd6e4f69fb1f1da2",
    }
    for name, expected in metadata_hashes.items():
        verify(folder / name, expected)
    return folder

#!/usr/bin/env python3
"""Create the environment, install packages, and download the local models.

    python3 setup.py
    python3 setup.py --hf-token YOUR_TOKEN

Then start the page with:

    .venv/bin/python stt_server.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Sequence

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def _ensure_venv() -> None:
    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return
    if not VENV_PYTHON.is_file():
        print(f"Creating {ROOT / '.venv'}", flush=True)
        venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


def _run(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.check_call(args)


def _cuda_torch_ok() -> bool:
    code = "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    return subprocess.call([sys.executable, "-c", code]) == 0


def install_packages() -> None:
    _run([sys.executable, "-m", "pip", "install", "-U", "pip"])
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchaudio",
            "--index-url",
            TORCH_INDEX,
        ]
    )
    _run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    if not _cuda_torch_ok():
        print("requirements.txt replaced the CUDA build of torch. Installing it again.", flush=True)
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "torch",
                "torchaudio",
                "--index-url",
                TORCH_INDEX,
            ]
        )
    if not _cuda_torch_ok():
        raise SystemExit("PyTorch cannot see a CUDA device after install.")


def _env_hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _snapshot(repo_id: str, dest: Path, token: str | None, allow_patterns: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {dest}", flush=True)
    kwargs: dict = {"repo_id": repo_id, "local_dir": str(dest)}
    if token:
        kwargs["token"] = token
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    try:
        snapshot_download(**kwargs)
    except TypeError:
        if token:
            kwargs.pop("token", None)
            kwargs["use_auth_token"] = token
        snapshot_download(**kwargs)


def _invalidate_hf_file(model_dir: Path, relative: str) -> None:
    target = model_dir / relative
    if target.is_file():
        target.unlink()
    meta = model_dir / ".cache" / "huggingface" / "download" / f"{relative}.metadata"
    if meta.is_file():
        meta.unlink()


def download_whisper(models_dir: Path, names: Sequence[str]) -> None:
    from stt_server import (
        WHISPER_REPO_IDS,
        expected_whisper_bin_size,
        is_whisper_model_dir,
        whisper_model_dir,
        whisper_weight_status,
    )

    for name in names:
        repo_id = WHISPER_REPO_IDS.get(name)
        dest = whisper_model_dir(models_dir, name)
        if repo_id is None:
            raise SystemExit(f"Unknown Whisper model {name!r}. Known sizes: {', '.join(WHISPER_REPO_IDS)}")
        if is_whisper_model_dir(dest):
            print(f"Already present: {dest}", flush=True)
            continue
        if whisper_weight_status(dest) and (dest / "model.bin").is_file():
            expected = expected_whisper_bin_size(dest)
            actual = (dest / "model.bin").stat().st_size
            detail = f"{actual:,} bytes"
            if expected:
                detail += f", expected {expected:,}"
            print(f"Incomplete download at {dest} ({detail}). Downloading again.", flush=True)
            _invalidate_hf_file(dest, "model.bin")
        _snapshot(repo_id, dest, token=None)
        if not is_whisper_model_dir(dest):
            status = whisper_weight_status(dest) or "folder is not a valid faster-whisper model"
            raise SystemExit(f"Download finished but {dest} is still unusable: {status}")


def download_pytorch_whisper(models_dir: Path) -> None:
    from stt_server import pytorch_whisper_dir, pytorch_whisper_ready

    dest = pytorch_whisper_dir(models_dir)
    if pytorch_whisper_ready(models_dir):
        print(f"Already present: {dest}", flush=True)
        return
    _snapshot(
        "openai/whisper-large-v3",
        dest,
        token=None,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )
    if not pytorch_whisper_ready(models_dir):
        raise SystemExit(f"Download finished but {dest} is missing model.safetensors or config.json.")


def download_diarization(models_dir: Path, token: str | None, *, legacy_31: bool) -> None:
    from stt_server import (
        PYANNOTE_LEGACY_REPO_IDS,
        PYANNOTE_PIPELINE_DIRNAME,
        PYANNOTE_PIPELINE_REPO,
        ensure_legacy_pipeline_local_paths,
        is_pyannote_pipeline_dir,
        pyannote_component_dir,
    )

    dest = pyannote_component_dir(models_dir, PYANNOTE_PIPELINE_DIRNAME)
    need_community = not is_pyannote_pipeline_dir(dest)
    need_legacy = False
    if legacy_31:
        for folder_name in PYANNOTE_LEGACY_REPO_IDS:
            legacy_dest = pyannote_component_dir(models_dir, folder_name)
            if not legacy_dest.is_dir() or not any(legacy_dest.iterdir()):
                need_legacy = True
                break
    if (need_community or need_legacy) and not token:
        raise SystemExit(
            "A Hugging Face token is required to download gated pyannote models.\n"
            "Accept https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "then run:\n"
            "  python setup.py --hf-token YOUR_TOKEN\n"
            "or set the HF_TOKEN environment variable."
        )
    if need_community:
        _snapshot(PYANNOTE_PIPELINE_REPO, dest, token=token)
        if not is_pyannote_pipeline_dir(dest):
            raise SystemExit(
                f"Download finished but {dest} is not a valid pyannote pipeline. "
                "Accept the community-1 license and retry."
            )
        print(f"Saved pyannote pipeline to {dest}", flush=True)
    else:
        print(f"Already present: {dest}", flush=True)

    if legacy_31:
        for folder_name, repo_id in PYANNOTE_LEGACY_REPO_IDS.items():
            legacy_dest = pyannote_component_dir(models_dir, folder_name)
            if legacy_dest.is_dir() and any(legacy_dest.iterdir()):
                print(f"Already present: {legacy_dest}", flush=True)
            else:
                _snapshot(repo_id, legacy_dest, token=token)
    if ensure_legacy_pipeline_local_paths(models_dir):
        print("Rewrote speaker-diarization-3.1 paths to this machine's models folder.", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install dependencies and download local speech models.")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated pyannote models.")
    parser.add_argument("--whisper", action="append", dest="whisper_models", metavar="NAME")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--legacy-3.1", action="store_true", dest="legacy_31")
    parser.add_argument("--skip-install", action="store_true", help="Only check and download models.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_venv()
    args = parse_args(argv)
    if not args.skip_install:
        install_packages()

    sys.path.insert(0, str(ROOT))
    from stt_server import configure_local_hub

    models_dir = args.models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    configure_local_hub(models_dir, offline=False)
    try:
        if not args.skip_whisper:
            download_whisper(models_dir, args.whisper_models or ["large-v3", "medium"])
            download_pytorch_whisper(models_dir)
        if not args.skip_diarization:
            download_diarization(models_dir, args.hf_token or _env_hf_token(), legacy_31=args.legacy_31)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print(f"Models ready in {models_dir}", flush=True)
    print("Start the page with: .venv/bin/python stt_server.py", flush=True)
    print("Then open http://127.0.0.1:8788", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

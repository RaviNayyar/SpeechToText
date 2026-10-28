#!/usr/bin/env python3
"""Local transcription server.

Open the printed URL, drop in a video or audio file, and download the transcript.
Models are read from ./models. Prepare them once with setup.py.
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

WHISPER_SUBDIR = "whisper"
PYANNOTE_SUBDIR = "pyannote"
HUB_CACHE_SUBDIR = "huggingface"
HF_HOME_SUBDIR = "hf-home"

# Local folder names under models/pyannote/ (must match setup.py).
# pyannote.audio 4.x requires community-1; 3.1 remains as a fallback layout.
DIARIZATION_PIPELINE_DIRS = (
    "speaker-diarization-community-1",
    "speaker-diarization-3.1",
)

WHISPER_REPO_IDS = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "medium": "Systran/faster-whisper-medium",
}

# Official Systran CTranslate2 float16 model.bin sizes. A shorter file is a
# truncated download (the Hub metadata can still claim the file is complete).
WHISPER_MODEL_BIN_SIZES = {
    "large-v3": 3_087_284_237,
    "medium": 1_527_906_378,
}

COMMUNITY_PIPELINE_FILES = (
    "config.yaml",
    "segmentation/pytorch_model.bin",
    "embedding/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
)

# Full pipeline repo used by pyannote.audio 4.x (segmentation/embedding/plda live inside).
PYANNOTE_PIPELINE_REPO = "pyannote/speaker-diarization-community-1"
PYANNOTE_PIPELINE_DIRNAME = "speaker-diarization-community-1"

# Older split 3.1 layout (only used when pyannote.audio 3.x is installed).
PYANNOTE_LEGACY_REPO_IDS = {
    "speaker-diarization-3.1": "pyannote/speaker-diarization-3.1",
    "segmentation-3.0": "pyannote/segmentation-3.0",
    "wespeaker-voxceleb-resnet34-LM": "pyannote/wespeaker-voxceleb-resnet34-LM",
}


def default_models_dir() -> Path:
    return DEFAULT_MODELS_DIR


def whisper_model_dir(models_dir: Path, name: str) -> Path:
    return models_dir / WHISPER_SUBDIR / name


def pyannote_component_dir(models_dir: Path, name: str) -> Path:
    return models_dir / PYANNOTE_SUBDIR / name


def hub_cache_dir(models_dir: Path) -> Path:
    return models_dir / HUB_CACHE_SUBDIR


def cached_blob_size(model_dir: Path, relative: str) -> int | None:
    """Return the size recorded by a previous Hugging Face snapshot, if any."""
    meta = model_dir / ".cache" / "huggingface" / "download" / f"{relative}.metadata"
    if not meta.is_file():
        return None
    try:
        commit = meta.read_text(encoding="utf-8").splitlines()[0].strip()
    except OSError:
        return None
    tree = model_dir / ".cache" / "huggingface" / "trees" / f"{commit}.json"
    if not tree.is_file():
        return None
    try:
        data = json.loads(tree.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    info = (data.get("files") or {}).get(relative) or {}
    size = info.get("lfs_size", info.get("size"))
    return int(size) if isinstance(size, int) else None


def expected_whisper_bin_size(model_dir: Path) -> int | None:
    cached = cached_blob_size(model_dir, "model.bin")
    if cached:
        return cached
    return WHISPER_MODEL_BIN_SIZES.get(model_dir.name)


def whisper_weight_status(model_dir: Path) -> str | None:
    """None when model.bin is complete. Otherwise a short reason it cannot be used."""
    weight = model_dir / "model.bin"
    if not weight.is_file():
        return f"missing {weight}"
    actual = weight.stat().st_size
    expected = expected_whisper_bin_size(model_dir)
    if expected is not None and actual != expected:
        return (
            f"incomplete {weight} ({actual:,} of {expected:,} bytes). "
            "Re-run: python setup.py"
        )
    if actual < 1_000_000:
        return f"{weight} is too small to be a Whisper model ({actual:,} bytes)"
    return None


def is_whisper_model_dir(path: Path) -> bool:
    """True if path is a complete CTranslate2 faster-whisper model folder."""
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    return whisper_weight_status(path) is None


def _pipeline_config(path: Path) -> dict | None:
    config_path = path / "config.yaml"
    if not path.is_dir() or not config_path.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("pipeline"), dict):
        return data
    return None


def is_pyannote_pipeline_dir(path: Path) -> bool:
    """True if path is a pipeline folder with the weights that config expects."""
    if _pipeline_config(path) is None:
        return False
    if path.name != PYANNOTE_PIPELINE_DIRNAME:
        return True
    for relative in COMMUNITY_PIPELINE_FILES:
        target = path / relative
        if not target.is_file():
            return False
        expected = cached_blob_size(path, relative)
        if expected is not None and target.stat().st_size != expected:
            return False
    return True


def ensure_legacy_pipeline_local_paths(models_dir: Path) -> bool:
    """Point speaker-diarization-3.1 at local folders when its paths are missing.

    Older downloads rewrote config.yaml with absolute paths from another
    machine (for example a Windows profile). Returns True when the file changed.
    """
    pipeline_dir = pyannote_component_dir(models_dir, "speaker-diarization-3.1")
    config_path = pipeline_dir / "config.yaml"
    data = _pipeline_config(pipeline_dir)
    if data is None:
        return False

    params = data.get("pipeline", {}).get("params")
    if not isinstance(params, dict):
        return False

    local_dirs = {
        "embedding": pyannote_component_dir(models_dir, "wespeaker-voxceleb-resnet34-LM"),
        "segmentation": pyannote_component_dir(models_dir, "segmentation-3.0"),
    }
    changed = False
    for key, local in local_dirs.items():
        if not local.is_dir():
            continue
        current = params.get(key)
        current_path = Path(str(current)).expanduser() if isinstance(current, str) else None
        if current_path is not None and current_path.is_dir():
            continue
        params[key] = local.resolve().as_posix()
        changed = True

    if not changed:
        return False

    import yaml

    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return True


def configure_local_hub(models_dir: Path, *, offline: bool) -> None:
    """Point Hugging Face caches at ./models and optionally forbid network access."""
    models_dir = models_dir.expanduser().resolve()
    cache = hub_cache_dir(models_dir)
    hf_home = models_dir / HF_HOME_SUBDIR
    cache.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    cache_str = str(cache)
    home_str = str(hf_home)
    os.environ["HF_HOME"] = home_str
    os.environ["HF_HUB_CACHE"] = cache_str
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_str
    os.environ["TRANSFORMERS_CACHE"] = cache_str
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def resolve_whisper_source(name_or_path: str, models_dir: Path) -> Path | None:
    """Return a local Whisper directory, or None if this name is not present locally."""
    raw = Path(name_or_path).expanduser()
    if raw.is_dir() and is_whisper_model_dir(raw):
        return raw.resolve()

    local = whisper_model_dir(models_dir, name_or_path)
    if is_whisper_model_dir(local):
        return local.resolve()
    return None


def list_local_diarization_pipelines(models_dir: Path) -> list[Path]:
    found: list[Path] = []
    pyannote_root = models_dir / PYANNOTE_SUBDIR
    for name in DIARIZATION_PIPELINE_DIRS:
        candidate = pyannote_root / name
        if is_pyannote_pipeline_dir(candidate):
            found.append(candidate.resolve())
    return found


def missing_models_hint(models_dir: Path, model_name: str | None = None) -> str:
    name = model_name or "large-v3"
    expected = whisper_model_dir(models_dir, name)
    pipeline = pyannote_component_dir(models_dir, PYANNOTE_PIPELINE_DIRNAME)
    lines = [
        f"Local models under {models_dir} are missing or incomplete.",
        "Repair or download them with:",
        "  python setup.py",
        "Pyannote weights are gated. Accept the license, then:",
        "  python setup.py --hf-token YOUR_TOKEN",
        "  https://huggingface.co/pyannote/speaker-diarization-community-1",
        f"Expected Whisper directory: {expected}",
        f"Expected diarization pipeline: {pipeline}",
    ]
    problem = whisper_weight_status(expected)
    if problem:
        lines.insert(1, problem)
    return "\n".join(lines)

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "large-v3"
DEFAULT_FALLBACK_MODEL = "medium"
SPEAKER_LABEL_RE = re.compile(r"^SPEAKER_\d+$")


class PipelineError(Exception):
    """Expected, user-facing failure (missing tools, bad input, gated model, etc.)."""


@dataclass(frozen=True)
class DeviceConfig:
    """Device strings for faster-whisper (CTranslate2) and pyannote (PyTorch)."""

    whisper_device: str
    diarize_device: str
    compute_type: str
    whisper_backend: str = "ctranslate2"


@dataclass(frozen=True)
class TimedWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class AsrSegment:
    start: float
    end: float
    text: str
    words: tuple[TimedWord, ...] = ()


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str


def _whisper_cuda_available() -> bool:
    """True when this CTranslate2 build can execute on an NVIDIA GPU."""
    try:
        import ctranslate2
    except ImportError:
        return False
    getter = getattr(ctranslate2, "get_cuda_device_count", None)
    if getter is None:
        return False
    try:
        return int(getter()) > 0
    except Exception:
        return False


def setup_device(models_dir: Path | None = None) -> DeviceConfig:
    """Detect GPU/CPU compute capability and return device string and precision."""
    try:
        import torch
    except ImportError as exc:
        raise PipelineError(
            "PyTorch is not installed. Install torch and torchaudio; see the module docstring."
        ) from exc

    if torch.cuda.is_available():
        if _whisper_cuda_available():
            return DeviceConfig(
                whisper_device="cuda",
                diarize_device="cuda",
                compute_type="float16",
                whisper_backend="ctranslate2",
            )
        if models_dir is not None and pytorch_whisper_ready(models_dir):
            return DeviceConfig(
                whisper_device="cuda",
                diarize_device="cuda",
                compute_type="float16",
                whisper_backend="pytorch",
            )
        print(
            "Speaker diarization will use the GPU. Speech recognition will use the CPU "
            "because this CTranslate2 build has no CUDA support.",
            flush=True,
        )
        return DeviceConfig(whisper_device="cpu", diarize_device="cuda", compute_type="int8")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        # faster-whisper/CTranslate2 does not support MPS; keep ASR on CPU/int8.
        return DeviceConfig(whisper_device="cpu", diarize_device="mps", compute_type="int8")

    return DeviceConfig(whisper_device="cpu", diarize_device="cpu", compute_type="int8")


def _register_ffmpeg_dll_dir(folder: Path) -> None:
    """Make FFmpeg shared libraries visible to ctypes/torchcodec on Windows."""
    folder_str = str(folder)
    current = os.environ.get("PATH", "")
    if folder_str.lower() not in current.lower().split(os.pathsep):
        os.environ["PATH"] = folder_str + os.pathsep + current
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(folder_str)
        except (FileNotFoundError, OSError):
            pass


def _windows_ffmpeg_bin_dirs() -> list[Path]:
    """Locate Gyan.FFmpeg even when this shell was opened before PATH was updated."""
    local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
    dirs: list[Path] = []
    packages = local_appdata / "Microsoft" / "WinGet" / "Packages"
    if packages.is_dir():
        for pkg in packages.glob("Gyan.FFmpeg*"):
            dirs.extend(path.parent for path in pkg.glob("**/ffmpeg.exe"))
    links = local_appdata / "Microsoft" / "WinGet" / "Links"
    if (links / "ffmpeg.exe").is_file():
        dirs.append(links)
    return dirs


def _require_ffmpeg() -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    if ffmpeg_bin and ffprobe_bin:
        _register_ffmpeg_dll_dir(Path(ffmpeg_bin).parent)
        return

    if os.name == "nt":
        for folder in _windows_ffmpeg_bin_dirs():
            ffmpeg = folder / "ffmpeg.exe"
            ffprobe = folder / "ffprobe.exe"
            if ffmpeg.is_file() and ffprobe.is_file():
                _register_ffmpeg_dll_dir(folder)
                if shutil.which("ffmpeg") and shutil.which("ffprobe"):
                    print(f"Using ffmpeg from {folder}", flush=True)
                    return

    missing = [name for name, path in (("ffmpeg", ffmpeg_bin), ("ffprobe", ffprobe_bin)) if path is None]
    raise PipelineError(
        f"{' and '.join(missing)} not found on PATH. Install ffmpeg and ensure both "
        "ffmpeg and ffprobe are available, then retry.\n"
        "  Windows: winget install Gyan.FFmpeg\n"
        "            Close and reopen the terminal so PATH picks it up.\n"
        "  macOS:   brew install ffmpeg\n"
        "  Ubuntu:  sudo apt-get install -y ffmpeg"
    )


def _import_ffmpeg():
    try:
        import ffmpeg
    except ImportError as exc:
        raise PipelineError(
            "The ffmpeg-python package is missing. Install dependencies with:\n"
            "  pip install ffmpeg-python"
        ) from exc
    return ffmpeg


def validate_input(input_path: Path) -> None:
    """Confirm the input exists and contains at least one audio stream."""
    if not input_path.exists():
        raise PipelineError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise PipelineError(f"Input path is not a file: {input_path}")

    _require_ffmpeg()
    ffmpeg = _import_ffmpeg()
    try:
        probe = ffmpeg.probe(str(input_path))
    except FileNotFoundError as exc:
        raise PipelineError(
            "ffmpeg/ffprobe could not be executed. Confirm ffmpeg is installed and on PATH."
        ) from exc
    except ffmpeg.Error as exc:
        detail = _ffmpeg_error_detail(exc)
        raise PipelineError(f"Could not read {input_path} as audio.\n{detail}") from exc
    except Exception as exc:
        raise PipelineError(f"Could not read {input_path} as audio: {exc}") from exc

    streams = probe.get("streams") or []
    if not any(stream.get("codec_type") == "audio" for stream in streams):
        raise PipelineError(f"No audio stream found in {input_path}")


def _ffmpeg_error_detail(exc: BaseException) -> str:
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, (bytes, bytearray)):
        text = stderr.decode("utf-8", errors="replace").strip()
        if text:
            return text
    return str(exc)


def convert_to_wav(input_path: Path, wav_path: Path) -> None:
    """Decode any ffmpeg-readable file to 16 kHz mono PCM WAV."""
    ffmpeg = _import_ffmpeg()
    try:
        (
            ffmpeg.input(str(input_path))
            .output(
                str(wav_path),
                ac=1,
                ar=SAMPLE_RATE,
                acodec="pcm_s16le",
                vn=None,
                loglevel="error",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except FileNotFoundError as exc:
        raise PipelineError(
            "ffmpeg was not found on PATH while converting audio. Install ffmpeg and retry."
        ) from exc
    except ffmpeg.Error as exc:
        detail = _ffmpeg_error_detail(exc)
        raise PipelineError(f"ffmpeg failed to convert {input_path} to WAV.\n{detail}") from exc

    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise PipelineError("Audio conversion produced an empty WAV file.")


def _release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _compute_type_candidates(device: str, preferred: str) -> list[str]:
    if device == "cuda":
        ordered = [preferred, "float16", "int8_float16", "int8", "float32"]
    else:
        ordered = [preferred, "int8", "int8_float32", "float32"]
    seen: set[str] = set()
    result: list[str] = []
    for item in ordered:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _load_whisper_model(
    model_name: str,
    fallback_model: str,
    whisper_device: str,
            compute_type: str,
    models_dir: Path,
):
    """Load faster-whisper from ./models, falling back across compute types and medium."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PipelineError(
            "faster-whisper is not installed. Install it with:\n"
            "  pip install faster-whisper"
        ) from exc

    sources: list[Path] = []
    for name in (model_name, fallback_model):
        if not name:
            continue
        resolved = resolve_whisper_source(name, models_dir)
        if resolved is None:
            if name == model_name:
                raise PipelineError(missing_models_hint(models_dir, name))
            print(
                f"Warning: fallback Whisper model {name!r} is not in {models_dir / 'whisper' / name}; skipping.",
                flush=True,
            )
            continue
        if resolved not in sources:
            sources.append(resolved)

    if not sources:
        raise PipelineError(missing_models_hint(models_dir, model_name))

    devices = [whisper_device]
    if whisper_device != "cpu":
        devices.append("cpu")

    last_error: BaseException | None = None
    for device in devices:
        device_compute = compute_type if device == whisper_device else "int8"
        for source in sources:
            for ctype in _compute_type_candidates(device, device_compute):
                print(f"Loading Whisper model {source} on {device} ({ctype})...", flush=True)
                try:
                    threads = min(16, os.cpu_count() or 4)
                    try:
                        model = WhisperModel(
                            str(source),
                            device=device,
                            compute_type=ctype,
                            cpu_threads=threads,
                            local_files_only=True,
                        )
                    except TypeError:
                        model = WhisperModel(
                            str(source),
                            device=device,
                            compute_type=ctype,
                            cpu_threads=threads,
                        )
                except Exception as exc:
                    last_error = exc
                    print(f"  load failed ({type(exc).__name__}: {exc})", flush=True)
                    _release_memory()
                    continue
                requested = resolve_whisper_source(model_name, models_dir)
                if (source, device, ctype) != (requested, whisper_device, compute_type):
                    print(
                        f"Using fallback configuration: model={source}, device={device}, compute_type={ctype}",
                        flush=True,
                    )
                return model
    raise PipelineError(
        "Could not load a local Whisper model. Tried "
        f"{[str(path) for path in sources]} on {devices}. Last error: {last_error}"
    ) from last_error


def _run_asr(
    wav_path: Path,
    model_name: str,
    fallback_model: str,
    whisper_device: str,
    compute_type: str,
    language: str | None,
    models_dir: Path,
    whisper_backend: str = "ctranslate2",
    on_progress=None,
) -> list[AsrSegment]:
    # Pass samples directly. faster-whisper would otherwise reopen the file
    # with PyAV, and current PyAV releases reject the metadata_errors argument.
    audio = _load_wav_mono_f32(wav_path)
    duration = len(audio) / SAMPLE_RATE
    if whisper_backend == "pytorch":

        if on_progress is not None:
            on_progress(0.08, "Loading Whisper on the GPU", None)
        return GpuWhisper.get(models_dir).transcribe(audio, language, on_progress)

    model = _load_whisper_model(
        model_name,
        fallback_model,
        whisper_device,
        compute_type,
        models_dir,
    )
    try:
        from faster_whisper import BatchedInferencePipeline

        runner = BatchedInferencePipeline(model=model)
        segments_iter, info = runner.transcribe(
            audio,
            batch_size=16 if whisper_device == "cuda" else 8,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True,
            language=language,
        )
        lang = getattr(info, "language", None)
        if lang:
            prob = getattr(info, "language_probability", None)
            extra = f" (p={prob:.2f})" if isinstance(prob, float) else ""
            print(f"Detected language: {lang}{extra}", flush=True)

        results: list[AsrSegment] = []
        for segment in segments_iter:
            text = (segment.text or "").strip()
            words: list[TimedWord] = []
            for word in segment.words or ():
                token = (word.word or "").strip()
                if not token:
                    continue
                start = float(word.start if word.start is not None else segment.start)
                end = float(word.end if word.end is not None else segment.end)
                if end < start:
                    end = start
                words.append(TimedWord(start=start, end=end, text=token))
            results.append(
                AsrSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                    words=tuple(words),
                )
            )
            if on_progress is not None and duration > 0 and text:
                on_progress(
                    0.08 + 0.72 * min(1.0, float(segment.end) / duration),
                    f"Transcribing {format_timestamp(segment.end)} of {format_timestamp(duration)}",
                    f"[{format_timestamp(segment.start)} -> {format_timestamp(segment.end)}] {text}",
                )
        return results
    finally:
        del model
        _release_memory()


def _call_from_pretrained(model_path: Path, cache_dir: Path | None = None):
    from pyannote.audio import Pipeline

    path = str(model_path)
    if cache_dir is not None:
        try:
            return Pipeline.from_pretrained(path, cache_dir=str(cache_dir))
        except TypeError:
            pass
    return Pipeline.from_pretrained(path)


def _load_diarization_pipeline(models_dir: Path):
    try:
        from pyannote.audio import Pipeline  # noqa: F401
    except ImportError as exc:
        raise PipelineError(
            "pyannote.audio is not installed. Install it with:\n"
            "  pip install 'pyannote.audio>=3.1.0'"
        ) from exc

    if ensure_legacy_pipeline_local_paths(models_dir):
        print("Rewrote speaker-diarization-3.1 paths to this machine's models folder.", flush=True)

    pipelines = list_local_diarization_pipelines(models_dir)
    if not pipelines:
        raise PipelineError(missing_models_hint(models_dir))

    cache_dir = hub_cache_dir(models_dir)
    try:
        import pyannote.audio as pyannote_audio

        pyannote_major = int(str(getattr(pyannote_audio, "__version__", "0")).split(".", 1)[0])
    except Exception:
        pyannote_major = 0

    last_error: BaseException | None = None
    for pipeline_dir in pipelines:
        if pipeline_dir.name == "speaker-diarization-3.1" and pyannote_major >= 4:
            print(
                "Skipping speaker-diarization-3.1 because pyannote.audio 4.x needs "
                "models/pyannote/speaker-diarization-community-1 instead.",
                flush=True,
            )
            continue
        print(f"Loading diarization pipeline from {pipeline_dir}...", flush=True)
        try:
            pipeline = _call_from_pretrained(pipeline_dir, cache_dir=cache_dir)
        except Exception as exc:
            last_error = exc
            print(f"  load failed ({type(exc).__name__}: {exc})", flush=True)
            continue
        if pipeline is None:
            last_error = RuntimeError(f"{pipeline_dir} returned None")
            continue
        return pipeline

    hint = missing_models_hint(models_dir)
    if last_error:
        raise PipelineError(f"{hint}\nLast error: {last_error}") from last_error
    raise PipelineError(hint)


def _move_pipeline_to_device(pipeline, device: str):
    import torch

    try:
        pipeline.to(torch.device(device))
        return device
    except Exception as exc:
        if device != "cpu":
            print(
                f"Warning: could not place diarization on {device} ({exc}); using CPU.",
                flush=True,
            )
            pipeline.to(torch.device("cpu"))
            return "cpu"
        raise


def _iter_diarization_turns(raw) -> Iterable[tuple[float, float, str]]:
    annotation = raw
    # community-1's exclusive diarization lines up more cleanly with ASR word times.
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        maybe = getattr(raw, attr, None)
        if maybe is not None and hasattr(maybe, "itertracks"):
            annotation = maybe
            break
    if not hasattr(annotation, "itertracks"):
        raise PipelineError(f"Unexpected diarization result type: {type(raw)!r}")
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        yield float(turn.start), float(turn.end), str(speaker)


def _normalize_speaker_id(raw: str, mapping: dict[str, str]) -> str:
    if raw in mapping:
        return mapping[raw]
    if SPEAKER_LABEL_RE.match(raw):
        mapping[raw] = raw
        return raw
    mapping[raw] = f"SPEAKER_{len(mapping):02d}"
    return mapping[raw]


def _load_wav_mono_f32(wav_path: Path):
    """Read the converted 16 kHz WAV as mono float32 samples in [-1, 1]."""
    import wave

    import numpy as np

    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)

    if sample_width != 2:
        raise PipelineError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit.")
    if n_frames <= 0:
        raise PipelineError("Converted WAV contains no audio frames.")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _load_wav_for_pyannote(wav_path: Path) -> dict:
    """Load our 16 kHz mono WAV as a tensor so pyannote does not need torchcodec."""
    import torch

    samples = _load_wav_mono_f32(wav_path)
    waveform = torch.from_numpy(samples.reshape(1, -1))
    return {"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": wav_path.stem}


def _run_diarization(
    wav_path: Path,
    diarize_device: str,
    models_dir: Path,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    pipeline=None,
    on_progress=None,
) -> list[SpeakerTurn]:
    owns_pipeline = pipeline is None
    if pipeline is None:
        pipeline = _load_diarization_pipeline(models_dir)
    try:
        _move_pipeline_to_device(pipeline, diarize_device)
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        if on_progress is not None:
            on_progress(0.82, "Identifying speakers", None)
        else:
            print("Running speaker diarization...", flush=True)
        audio = _load_wav_for_pyannote(wav_path)

        def _hook(step_name, step_artifact, file=None, total=None, completed=None):
            if on_progress is None or not total or completed is None:
                return
            on_progress(
                0.82 + 0.16 * (float(completed) / float(total)),
                f"Identifying speakers ({step_name})",
                None,
            )

        try:
            raw = pipeline(audio, hook=_hook, **kwargs)
        except TypeError:
            raw = pipeline(audio, **kwargs) if kwargs else pipeline(audio)

        mapping: dict[str, str] = {}
        turns: list[SpeakerTurn] = []
        for start, end, speaker in _iter_diarization_turns(raw):
            if end <= start:
                continue
            turns.append(
                SpeakerTurn(
                    start=start,
                    end=end,
                    speaker=_normalize_speaker_id(speaker, mapping),
                )
            )
        turns.sort(key=lambda item: (item.start, item.end))
        return turns
    finally:
        if owns_pipeline:
            del pipeline
            _release_memory()


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _speaker_for_span(start: float, end: float, turns: Sequence[SpeakerTurn]) -> str:
    if not turns:
        return "SPEAKER_00"

    best_speaker = turns[0].speaker
    best_overlap = 0.0
    for turn in turns:
        amount = _overlap(start, end, turn.start, turn.end)
        if amount > best_overlap:
            best_overlap = amount
            best_speaker = turn.speaker
    if best_overlap > 0.0:
        return best_speaker

    midpoint = (start + end) / 2.0
    best_dist = float("inf")
    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return turn.speaker
        dist = min(abs(midpoint - turn.start), abs(midpoint - turn.end))
        if dist < best_dist:
            best_dist = dist
            best_speaker = turn.speaker
    return best_speaker


def _clean_text(parts: Iterable[str]) -> str:
    joined = " ".join(part.strip() for part in parts if part and part.strip())
    return re.sub(r"\s+", " ", joined).strip()


def _assign_speakers(
    asr_segments: Sequence[AsrSegment],
    turns: Sequence[SpeakerTurn],
    max_gap: float = 0.8,
) -> list[TranscriptSegment]:
    """Align ASR timestamps with diarization turns at word level when possible."""
    words: list[tuple[TimedWord, str]] = []
    for segment in asr_segments:
        if segment.words:
            for word in segment.words:
                speaker = _speaker_for_span(word.start, word.end, turns)
                words.append((word, speaker))
        elif segment.text:
            speaker = _speaker_for_span(segment.start, segment.end, turns)
            words.append((TimedWord(segment.start, segment.end, segment.text), speaker))

    if not words:
        return []

    grouped: list[TranscriptSegment] = []
    current_words = [words[0][0]]
    current_speaker = words[0][1]
    for word, speaker in words[1:]:
        gap = word.start - current_words[-1].end
        same_run = speaker == current_speaker and gap <= max_gap
        if same_run:
            current_words.append(word)
            continue
        text = _clean_text(item.text for item in current_words)
        if text:
            grouped.append(
                TranscriptSegment(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    speaker=current_speaker,
                    text=text,
                )
            )
        current_words = [word]
        current_speaker = speaker

    text = _clean_text(item.text for item in current_words)
    if text:
        grouped.append(
            TranscriptSegment(
                start=current_words[0].start,
                end=current_words[-1].end,
                speaker=current_speaker,
                text=text,
            )
        )
    return grouped


def transcribe_and_diarize(
    input_path: Path,
    device: DeviceConfig,
    compute_type: str | None = None,
    *,
    model_name: str = DEFAULT_MODEL,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    language: str | None = None,
    models_dir: Path | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    on_progress=None,
    diarization_pipeline=None,
) -> list[TranscriptSegment]:
    """Run transcription and diarization, aligning timestamps with speaker turns."""

    def report(fraction: float, message: str, preview: str | None = None) -> None:
        if on_progress is not None:
            on_progress(fraction, message, preview)
            return
        print(message, flush=True)
        if preview:
            print(preview, flush=True)

    resolved_models = (models_dir or default_models_dir()).expanduser().resolve()
    configure_local_hub(resolved_models, offline=True)
    effective_compute = compute_type or device.compute_type
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        tmp_dir = tempfile.TemporaryDirectory(prefix="asr-diarize-")
        wav_path = Path(tmp_dir.name) / "audio_16k_mono.wav"
        report(0.02, f"Converting {input_path.name}")
        convert_to_wav(input_path, wav_path)

        report(0.06, "Running speech recognition")
        asr_segments = _run_asr(
            wav_path,
            model_name=model_name,
            fallback_model=fallback_model,
            whisper_device=device.whisper_device,
            compute_type=effective_compute,
            language=language,
            models_dir=resolved_models,
            whisper_backend=device.whisper_backend,
            on_progress=report,
        )
        if not asr_segments:
            print("Warning: Whisper produced no speech segments.", flush=True)
            return []

        turns = _run_diarization(
            wav_path,
            diarize_device=device.diarize_device,
            models_dir=resolved_models,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            pipeline=diarization_pipeline,
            on_progress=report,
        )
        if not turns:
            print("Warning: no speaker turns detected; defaulting to SPEAKER_00.", flush=True)

        report(0.98, "Aligning speakers with the transcript")
        return _assign_speakers(asr_segments, turns)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def format_timestamp(seconds: float) -> str:
    """Format a time position as HH:MM:SS (integer seconds)."""
    total = int(round(max(0.0, float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_output(segments: Sequence[TranscriptSegment], output_path: Path) -> None:
    """Write transcript lines atomically so a crash cannot leave a truncated file."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"[{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}] {seg.speaker}: {seg.text}"
        for seg in segments
    ]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    fd, tmp_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, output_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise



HF_WHISPER_DIRNAME = "large-v3"
BATCH_SIZE = 16
_STAMP = re.compile(r"<\|([^|]+)\|>")


def pytorch_whisper_dir(models_dir: Path) -> Path:
    return models_dir / "whisper-hf" / HF_WHISPER_DIRNAME


def pytorch_whisper_ready(models_dir: Path) -> bool:
    folder = pytorch_whisper_dir(models_dir)
    return (folder / "model.safetensors").is_file() and (folder / "config.json").is_file()


class GpuWhisper:
    """One resident Whisper model. Loading it is slow, so the process keeps it."""

    _instance: GpuWhisper | None = None
    _lock = threading.Lock()

    def __init__(self, models_dir: Path):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        folder = pytorch_whisper_dir(models_dir)
        if not pytorch_whisper_ready(models_dir):
            raise FileNotFoundError(f"PyTorch Whisper weights are not in {folder}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            folder,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.model.to("cuda")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, models_dir: Path) -> GpuWhisper:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(models_dir)
            return cls._instance

    def transcribe(self, audio: np.ndarray, language: str | None, on_progress) -> list[AsrSegment]:
        """Transcribe mono 16 kHz audio. on_progress(fraction, message, preview_line)."""
        import torch
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        total = max(len(audio) / SAMPLE_RATE, 0.001)
        speech = get_speech_timestamps(
            audio,
            VadOptions(max_speech_duration_s=30, min_silence_duration_ms=500),
        )
        pieces = [(int(item["start"]), audio[int(item["start"]) : int(item["end"])]) for item in speech]
        pieces = [(start, chunk) for start, chunk in pieces if len(chunk) >= SAMPLE_RATE // 5]
        if not pieces:
            return []

        segments: list[AsrSegment] = []
        with self._infer_lock, torch.inference_mode():
            for batch_start in range(0, len(pieces), BATCH_SIZE):
                group = pieces[batch_start : batch_start + BATCH_SIZE]
                new_segments = self._transcribe_batch(group, language)
                segments.extend(new_segments)
                done = min(total, (group[-1][0] + len(group[-1][1])) / SAMPLE_RATE)
                fraction = 0.08 + 0.72 * (done / total)
                message = f"Transcribing {format_timestamp(done)} of {format_timestamp(total)}"
                if on_progress is not None:
                    if not new_segments:
                        on_progress(fraction, message, None)
                    for segment in new_segments:
                        on_progress(fraction, message, _preview_line(segment))
        return segments

    def _transcribe_batch(self, group: list[tuple[int, np.ndarray]], language: str | None) -> list[AsrSegment]:
        import torch

        chunks = [chunk for _, chunk in group]
        features = self.processor.feature_extractor(
            chunks,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features
        features = features.to("cuda", dtype=torch.float16)
        kwargs = {"task": "transcribe", "num_beams": 1, "return_timestamps": True}
        if language:
            kwargs["language"] = language
        generated = self.model.generate(features, **kwargs)
        sequences = generated["sequences"] if isinstance(generated, dict) else generated
        decoded = self.processor.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=False,
            decode_with_timestamps=True,
        )
        segments: list[AsrSegment] = []
        for (start, _), text in zip(group, decoded):
            segments.extend(_segments_from_stamps(text, start / SAMPLE_RATE))
        return segments


def _preview_line(segment: AsrSegment) -> str:
    return f"[{format_timestamp(segment.start)} -> {format_timestamp(segment.end)}] {segment.text}"


def _segments_from_stamps(decoded: str, offset: float) -> list[AsrSegment]:
    """Turn Whisper timestamp tokens into timed phrases, relative to a chunk offset."""
    parts = _STAMP.split(decoded)
    pending_start: float | None = None
    pending_text: list[str] = []
    words: list[TimedWord] = []

    def flush(end: float) -> None:
        nonlocal pending_start, pending_text
        if pending_start is None:
            return
        text = "".join(pending_text).strip()
        pending_text = []
        start = pending_start
        pending_start = None
        if not text:
            return
        if end < start:
            end = start
        words.append(TimedWord(start=start + offset, end=end + offset, text=text))

    for index, part in enumerate(parts):
        if index % 2 == 0:
            if pending_start is not None:
                pending_text.append(part)
            continue
        if part in {"endoftext", "startoftranscript"}:
            flush(pending_start or 0.0)
            pending_start = None
            continue
        try:
            stamp = float(part)
        except ValueError:
            continue
        if pending_start is None:
            pending_start = stamp
            continue
        flush(stamp)
        pending_start = stamp
    return _group_words(words)


def _group_words(words: list[TimedWord], max_gap: float = 0.8) -> list[AsrSegment]:
    groups: list[list[TimedWord]] = [[words[0]]]
    for word in words[1:]:
        if word.start - groups[-1][-1].end <= max_gap:
            groups[-1].append(word)
        else:
            groups.append([word])
    segments: list[AsrSegment] = []
    for group in groups:
        text = " ".join(word.text for word in group).strip()
        if not text:
            continue
        segments.append(
            AsrSegment(
                start=group[0].start,
                end=group[-1].end,
                text=text,
                words=tuple(group),
            )
        )
    return segments

PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transcribe</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1c1915;
      --muted: #6b645c;
      --line: #d9d2c8;
      --paper: #f6f3ee;
      --card: #fffdf9;
      --accent: #0f6e56;
      --accent-ink: #ffffff;
      --danger: #8f2d2d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--paper);
      color: var(--ink);
      font: 16px/1.5 "Segoe UI", system-ui, sans-serif;
    }
    main {
      width: min(760px, calc(100% - 32px));
      margin: 0 auto;
      padding: 48px 0 64px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 32px;
      font-weight: 650;
      letter-spacing: -0.03em;
    }
    .lede { margin: 0 0 28px; color: var(--muted); }
    .drop {
      display: grid;
      place-items: center;
      gap: 8px;
      min-height: 180px;
      padding: 28px;
      border: 1.5px dashed var(--line);
      border-radius: 16px;
      background: var(--card);
      text-align: center;
      cursor: pointer;
    }
    .drop.hot { border-color: var(--accent); background: #f3faf7; }
    .drop strong { font-size: 18px; }
    .drop span { color: var(--muted); }
    .browse {
      margin-top: 4px;
      border: 0;
      border-radius: 999px;
      padding: 8px 14px;
      background: var(--ink);
      color: white;
      font: inherit;
      cursor: pointer;
    }
    input[type="file"] { display: none; }
    .status { margin-top: 22px; }
    .status p { margin: 0 0 8px; }
    .bar {
      height: 8px;
      border-radius: 999px;
      background: #e7e1d8;
      overflow: hidden;
    }
    .bar > div {
      height: 100%;
      width: 0;
      background: var(--accent);
      transition: width 0.2s ease;
    }
    .transcript {
      margin-top: 22px;
      padding: 16px 18px;
      border-radius: 16px;
      background: var(--card);
      border: 1px solid var(--line);
      white-space: pre-wrap;
      min-height: 120px;
    }
    .actions { margin-top: 16px; display: flex; gap: 10px; }
    a.download, button.reset {
      display: inline-block;
      text-decoration: none;
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      cursor: pointer;
    }
    a.download { background: var(--accent); color: var(--accent-ink); }
    button.reset { background: transparent; border: 1px solid var(--line); color: var(--ink); }
    .error { color: var(--danger); }
    .hidden { display: none; }
    footer { margin-top: 28px; color: var(--muted); font-size: 14px; }
  </style>
</head>
<body>
  <main>
    <h1>Transcribe</h1>
    <p class="lede">Drop a video or audio file. The soundtrack is transcribed with speaker labels, and you can download the text when it is ready.</p>
    <div class="drop" id="drop">
      <strong>Drop a recording here</strong>
      <span>or choose a file from this computer</span>
      <button class="browse" id="browse" type="button">Browse</button>
      <input id="file" type="file" accept="audio/*,video/*,.mp3,.wav,.m4a,.flac,.ogg,.aac,.mp4,.mov,.mkv,.webm,.m4v">
    </div>
    <section class="status hidden" id="status">
      <p id="message">Starting</p>
      <div class="bar" aria-hidden="true"><div id="fill"></div></div>
    </section>
    <pre class="transcript hidden" id="transcript"></pre>
    <div class="actions hidden" id="actions">
      <a class="download" id="download" href="#">Download transcript</a>
      <button class="reset" id="reset" type="button">Transcribe another</button>
    </div>
    <p class="error hidden" id="error"></p>
    <footer id="device">Preparing the speech models…</footer>
  </main>
  <script>
    const drop = document.querySelector("#drop");
    const fileInput = document.querySelector("#file");
    const browse = document.querySelector("#browse");
    const statusEl = document.querySelector("#status");
    const messageEl = document.querySelector("#message");
    const fill = document.querySelector("#fill");
    const transcript = document.querySelector("#transcript");
    const actions = document.querySelector("#actions");
    const download = document.querySelector("#download");
    const errorEl = document.querySelector("#error");
    const deviceEl = document.querySelector("#device");
    const reset = document.querySelector("#reset");
    let polling = 0;

    drop.addEventListener("click", (event) => {
      if (event.target === browse) return;
      fileInput.click();
    });
    browse.addEventListener("click", (event) => {
      event.preventDefault();
      fileInput.click();
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) start(fileInput.files[0]);
    });
    drop.addEventListener("dragover", (event) => {
      event.preventDefault();
      drop.classList.add("hot");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("hot");
      const file = event.dataTransfer.files[0];
      if (file) start(file);
    });
    reset.addEventListener("click", () => {
      window.clearTimeout(polling);
      fileInput.value = "";
      statusEl.classList.add("hidden");
      transcript.classList.add("hidden");
      actions.classList.add("hidden");
      errorEl.classList.add("hidden");
      transcript.textContent = "";
      drop.classList.remove("hidden");
    });

    async function refreshHealth() {
      try {
        const health = await fetch("/api/health").then((response) => response.json());
        if (health.error) {
          deviceEl.textContent = health.error;
          return;
        }
        if (health.ready) {
          deviceEl.textContent = "Ready. " + (health.device || "");
          return;
        }
        deviceEl.textContent = "Preparing the speech models…";
        window.setTimeout(refreshHealth, 1000);
      } catch (err) {
        deviceEl.textContent = "The page cannot reach the transcription program.";
      }
    }
    refreshHealth();

    async function start(file) {
      errorEl.classList.add("hidden");
      actions.classList.add("hidden");
      drop.classList.add("hidden");
      statusEl.classList.remove("hidden");
      transcript.classList.remove("hidden");
      transcript.textContent = "";
      messageEl.textContent = "Uploading " + file.name;
      fill.style.width = "2%";
      const body = new FormData();
      body.append("file", file);
      const response = await fetch("/api/jobs", { method: "POST", body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        showError(payload.detail || "Upload failed.");
        return;
      }
      watch(payload.id);
    }

    async function watch(id) {
      const response = await fetch("/api/jobs/" + id);
      const job = await response.json();
      messageEl.textContent = job.message || job.status;
      fill.style.width = Math.round((job.progress || 0) * 100) + "%";
      transcript.textContent = (job.lines || []).join("\n");
      transcript.classList.toggle("hidden", !(job.lines || []).length);
      if (job.status === "done") {
        download.href = "/api/jobs/" + id + "/download";
        download.download = job.filename.replace(/\.[^.]+$/, "") + ".txt";
        actions.classList.remove("hidden");
        return;
      }
      if (job.status === "error") {
        showError(job.error || job.message || "Transcription failed.");
        return;
      }
      polling = window.setTimeout(() => watch(id), 700);
    }

    function showError(text) {
      errorEl.textContent = text;
      errorEl.classList.remove("hidden");
      drop.classList.remove("hidden");
    }
  </script>
</body>
</html>
'''


HOST = "0.0.0.0"
PORT = 8788
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work" / "jobs"

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_models_ready = threading.Event()
_load_error: str | None = None
_device = None
_diarization = None


def _safe_name(name: str) -> str:
    base = Path(name or "recording").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip("._ ")
    return (cleaned or "recording")[:120]


def _preload() -> None:
    global _load_error, _device, _diarization
    try:
        models_dir = default_models_dir()
        _device = setup_device(models_dir)
        if _device.whisper_backend == "pytorch":
            GpuWhisper.get(models_dir)
        _diarization = _load_diarization_pipeline(models_dir)
        _move_pipeline_to_device(_diarization, _device.diarize_device)
    except Exception as exc:
        _load_error = str(exc)
        traceback.print_exc()
    finally:
        _models_ready.set()


def _public_job(job: dict) -> dict:
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "filename": job["filename"],
        "lines": list(job["lines"]),
        "error": job["error"],
    }


def _run_job(job_id: str, media: Path, language: str | None) -> None:
    job = _jobs[job_id]

    def on_progress(fraction: float, message: str, preview: str | None = None) -> None:
        with _lock:
            job["progress"] = max(0.0, min(0.99, float(fraction)))
            job["message"] = message
            if preview:
                job["lines"].append(preview)

    try:
        _models_ready.wait()
        if _load_error or _device is None:
            raise PipelineError(_load_error or "Models failed to load.")
        with _lock:
            job["status"] = "running"
            job["message"] = "Starting"
        segments = transcribe_and_diarize(
            media,
            _device,
            language=language,
            on_progress=on_progress,
            diarization_pipeline=_diarization,
        )
        lines = [
            f"[{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}] {seg.speaker}: {seg.text}"
            for seg in segments
        ]
        transcript = job["dir"] / "transcript.txt"
        transcript.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
        with _lock:
            job["lines"] = lines
            job["progress"] = 1.0
            job["status"] = "done"
            job["message"] = f"Ready. {len(lines)} lines."
            job["transcript"] = transcript
    except Exception as exc:
        traceback.print_exc()
        with _lock:
            job["status"] = "error"
            job["error"] = str(exc)
            job["message"] = str(exc)
    finally:
        try:
            media.unlink()
        except OSError:
            pass


def main() -> None:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
    import uvicorn

    @asynccontextmanager
    async def lifespan(_app):
        threading.Thread(target=_preload, name="load-models", daemon=True).start()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/health")
    def health() -> dict:
        ready = _models_ready.is_set() and _load_error is None and _device is not None
        device = None
        if _device is not None and _device.whisper_device == "cuda" and _device.diarize_device == "cuda":
            device = "Speech recognition and speaker labels both run on the GPU."
        elif _device is not None:
            device = f"Speech recognition: {_device.whisper_device}. Speaker labels: {_device.diarize_device}."
        return {"ready": ready, "loading": not _models_ready.is_set(), "error": _load_error, "device": device}

    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse(PAGE)

    @app.post("/api/jobs")
    async def create_job(file: UploadFile = File(...), language: str = Form("")) -> dict:
        with _lock:
            if any(job["status"] in {"queued", "running"} for job in _jobs.values()):
                raise HTTPException(status_code=409, detail="A transcription is already running.")
            job_id = uuid.uuid4().hex
            folder = WORK / job_id
            folder.mkdir(parents=True, exist_ok=True)
            filename = _safe_name(file.filename or "recording")
            job = {
                "id": job_id,
                "status": "queued",
                "progress": 0.0,
                "message": "Waiting for the models" if not _models_ready.is_set() else "Queued",
                "filename": filename,
                "lines": [],
                "error": None,
                "dir": folder,
                "transcript": None,
            }
            _jobs[job_id] = job

        media = folder / filename
        with media.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        chosen = language.strip() or None
        threading.Thread(target=_run_job, args=(job_id, media, chosen), name=f"job-{job_id[:8]}", daemon=True).start()
        return _public_job(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown job.")
            return _public_job(job)

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str) -> FileResponse:
        with _lock:
            job = _jobs.get(job_id)
            transcript = None if job is None else job.get("transcript")
            download_name = "transcript.txt" if job is None else f"{Path(job['filename']).stem}.txt"
        if job is None or transcript is None or not Path(transcript).is_file():
            raise HTTPException(status_code=404, detail="Transcript is not ready.")
        return FileResponse(transcript, media_type="text/plain; charset=utf-8", filename=download_name)

    print(f"Open http://127.0.0.1:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()

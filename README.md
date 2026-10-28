# Speech to text

Offline transcription in the browser. Drop a video or audio file on the page. Video files use the soundtrack. The finished download labels each line with a speaker.

## Setup

ffmpeg must be on `PATH`. Then, from this folder:

```bash
python3 setup.py
```

`setup.py` creates `.venv`, installs the CUDA build of PyTorch plus `requirements.txt`, and checks the models under `models/`. Whisper weights are public. Speaker diarization uses a gated pyannote model. If that model is not already in `models/pyannote`, accept the license and run:

```bash
python3 setup.py --hf-token YOUR_TOKEN
```

License page: https://huggingface.co/pyannote/speaker-diarization-community-1

## Run

```bash
.venv/bin/python stt_server.py
```

Open http://127.0.0.1:8788. One file runs at a time. Progress lines appear while speech recognition is still going. When diarization finishes, every line is replaced with the speaker-labeled transcript, and the download button saves that text:

```text
[00:01:12 -> 00:01:18] SPEAKER_01: hello there
```

Speech recognition and speaker labels both stay on the GPU when the CUDA weights are present. Models stay on disk and in memory between jobs.

# LocalTranscribe

A small **offline speech-to-text tool** for long lecture recordings, built around [Microsoft VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR), tuned to run on a laptop GPU with only **8 GB of VRAM**.

Audio never leaves the machine: decoding uses a local FFmpeg, the model loads in offline mode, and the Gradio web UI only listens on `127.0.0.1`.

## Features

- Web UI: pick a file or paste a local path, add context or domain terms, try 1 minute first, then run the whole file
- Splits audio into 20/30/60-second chunks, saving each one as soon as it finishes; a crashed run can resume from a given second
- Exports `transcript.txt` (timestamps + speaker IDs), `transcript.srt` and structured `transcript.json`
- Background start/stop scripts for Windows (`Start.cmd`, `Stop.cmd`)

## Fitting a 7B-class ASR model into 8 GB

`engine.py` does the following:

- **NF4 4-bit quantization** of the language-model linear layers (bitsandbytes); audio tokenizers/connectors stay in BF16
- **2-second streaming audio encoding**, aligned to the encoder stride (48 000 samples = 15 × 3200), so no latent frames are lost at boundaries
- A forward pre-hook on `lm_head` that keeps only the last-token logits, so the full prompt × vocabulary matrix is never materialized
- Frees the CUDA cache after every forward pass, which avoids Windows WDDM paging into shared memory

Measured model footprint: ~6.4 GiB VRAM on an RTX 3070 Laptop GPU.

## Setup

```powershell
python -m venv runtime
runtime\Scripts\pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
runtime\Scripts\pip install -r requirements-lock.txt
git clone https://github.com/microsoft/VibeVoice vendor/VibeVoice
# download microsoft/VibeVoice-ASR to models/VibeVoice-ASR
# and the Qwen2.5 tokenizer files to models/QwenTokenizer
Start.cmd
```

CLI usage: `python engine.py audio.m4a --limit 60 --context "German lecture, Elektrotechnik"`

## Limitations

- Speaker IDs are only valid within one chunk
- Chunk boundaries can cut sentences in half
- Quantized, chunked output is not equivalent to the official full-precision evaluation; check technical terms by listening again

## Credits

Model and inference code: [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice). This repository only contains the wrapper, memory optimizations and UI.

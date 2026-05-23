<div align="center">
  <img src="./assets/AI-ImagerForgeLogo.png" alt="AI-ImageForge logo" width="180"/>

  # AI-ImageForge

  A recoverable text-to-image demo with Hugging Face generation and local fallback rendering.
</div>

## Why This Version Works Better

Hackathon image-generation demos often fail for reasons that are not really product failures:
network latency, model cold starts, expired tokens, heavyweight installs, or a UI that promises
controls the backend does not use. This version is designed to stay presentable when those things
happen.

- Demo mode works without a token, network, GPU, or provider credits.
- Live mode uses Hugging Face `InferenceClient.text_to_image` when a token is available.
- Negative prompt, seed, model, batch size, quality, width, height, steps, and guidance are wired into generation.
- If live generation fails, the app falls back to a local deterministic renderer instead of stopping the flow.
- Default dependencies are light enough for a judging laptop.

## Features

- Streamlit interface for prompt writing, model choice, style, quality, seed, and batch generation.
- Hugging Face Inference Providers with automatic provider routing.
- Local procedural fallback renderer for demo resilience.
- Small in-session gallery of recent successful generations.
- Basic hardware detection for CPU, NVIDIA CUDA, and AMD DirectML.

## Quick Start

Requires Python 3.9 or newer.

```bash
cd web-genAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

On Windows, you can also run:

```bat
start.bat
```

On macOS or Linux:

```bash
./start.sh
```

## Live Generation

1. Create a Hugging Face token at `https://huggingface.co/settings/tokens`.
2. Paste it into the sidebar token field, or set it before launching:

```bash
export HF_TOKEN=hf_your_token_here
streamlit run app.py
```

Turn off Demo mode in the sidebar to use live generation. Leave Demo mode on for rehearsals,
offline judging, or development.

## Models

The app currently exposes:

- `black-forest-labs/FLUX.1-schnell`
- `stabilityai/stable-diffusion-xl-base-1.0`
- `stable-diffusion-v1-5/stable-diffusion-v1-5`
- `prompthero/openjourney`

Provider availability can change, so fallback behavior is part of the product rather than an error.

## Project Layout

```text
app.py              Streamlit application and generation logic
launcher.py         Cross-platform Python launcher
start.sh            macOS/Linux setup and launch script
start.bat           Windows setup and launch script
requirements.txt    Minimal runtime dependencies
pyproject.toml      Project metadata and optional extras
assets/             Logo and static assets
```

## Optional Hardware Packages

The application does not require local diffusion inference. If you want additional hardware probing,
install optional extras:

```bash
pip install ".[nvidia]"
pip install ".[amd]"
```

The default path remains API-first because it is faster to install and easier to demo reliably.

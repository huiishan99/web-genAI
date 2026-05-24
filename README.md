<div align="center">
  <img src="./assets/AI-ImagerForgeLogo.png" alt="AI-ImageForge logo" width="180"/>

  # AI-ImageForge

  A low-cost text-to-image workbench with Hugging Face generation, explicit sketch mode,
  and launch-safe deployment controls.
</div>

## Why This Version Works Better

Hackathon image-generation demos often fail for reasons that are not really product failures:
network latency, model cold starts, expired tokens, heavyweight installs, or a UI that promises
controls the backend does not use. This version is designed to stay presentable when those things
happen.

- Sketch mode works without a token, network, GPU, or provider credits.
- Live mode uses Hugging Face `InferenceClient.text_to_image` when a token is available.
- Negative prompt, seed, model, batch size, quality, width, height, steps, and guidance are wired into generation.
- Public deployments can use a server-side `HF_TOKEN`, a session-only user token, or Sketch mode.
- Production deployments can disable live fallback so demo images are never mistaken for real AI output.
- Default dependencies are light enough for a judging laptop.

## Features

- Streamlit interface for prompt writing, model choice, style, quality, seed, and batch generation.
- Hugging Face Inference Providers with automatic provider routing.
- Local procedural Sketch mode for free previews and offline resilience.
- Launch guard status for deployment profile, token source, and live fallback behavior.
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

## Low-Cost Product Modes

The app is designed to look product-ready without forcing the owner to pay for every public visitor.

- `Sketch mode`: free local renderer. Useful for portfolio visitors, rehearsals, and zero-cost demos.
- `Live provider` with `HF_TOKEN`: real Hugging Face image generation paid by the configured account.
- `Live provider` with a session token: real generation paid by the user who enters their own token.

For a public deployment where you do not want surprise charges, leave `HF_TOKEN` unset and keep
`ALLOW_SESSION_TOKENS=true`. Visitors can still explore Sketch mode, and power users can bring
their own Hugging Face token for real output.

## Live Generation

1. Create a Hugging Face token at `https://huggingface.co/settings/tokens`.
2. Paste it into `.streamlit/secrets.toml` for local development, or configure it as a deployment secret:

```toml
HF_TOKEN = "hf_your_token_here"
```

You can also set it for a one-off terminal run:

```bash
export HF_TOKEN=hf_your_token_here
streamlit run app.py
```

Turn off Sketch mode in the sidebar to use live generation. Leave Sketch mode on for rehearsals,
offline judging, development, or no-cost public browsing.

Optional launch controls:

```bash
export IMAGEFORGE_PROFILE=production
export ALLOW_LIVE_FALLBACK=false
export ALLOW_SESSION_TOKENS=true
export ALLOW_DEMO_MODE=true
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for a practical no/low-cost deployment path.

## Testing Generation

Run a no-cost smoke test first. It writes a real PNG produced by Sketch mode:

```bash
.venv/bin/python scripts/smoke_generation.py
```

Then test live generation with fallback disabled. This proves the image came from Hugging Face:

```bash
HF_TOKEN=hf_your_token_here .venv/bin/python scripts/smoke_generation.py --live
```

The smoke test writes images into `outputs/`, which is ignored by git.

In the web UI, live generation requires the same setup: enter a session token or configure `HF_TOKEN`,
turn off Sketch mode, click Generate, and confirm the result source is `Hugging Face`.

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
scripts/            Smoke tests and maintenance scripts
```

## Optional Hardware Packages

The application does not require local diffusion inference. If you want additional hardware probing,
install optional extras:

```bash
pip install ".[nvidia]"
pip install ".[amd]"
```

The default path remains API-first because it is faster to install and easier to demo reliably.

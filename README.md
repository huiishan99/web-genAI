<div align="center">
  <img src="./assets/AI-ImagerForgeLogo.png" alt="AI-ImageForge logo" width="180"/>

  # AI-ImageForge

  A low-cost text-to-image workbench with Hugging Face generation, explicit sketch mode,
  and launch-safe deployment controls.
</div>

![AI-ImageForge live generation UI](./assets/readme-live-generation.png)

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

## Quick Start on macOS

Requires Python 3.9 or newer.

```bash
cd web-genAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

To stop the app, press `Control+C` in the terminal that is running Streamlit. If a background
process is already using the port, find and stop it:

```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
kill <PID>
```

## Quick Start on Windows

Use the bundled launcher:

```bat
start.bat
```

Or run the commands manually in PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Quick Start on Linux

```bash
cd web-genAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Low-Cost Product Modes

The app is designed to look product-ready without forcing the owner to pay for every public visitor.

- `Sketch mode`: free local renderer. Useful for portfolio visitors, rehearsals, and zero-cost demos.
- `Live provider` with `HF_TOKEN`: real Hugging Face image generation paid by the configured account.
- `Live provider` with a session token: real generation paid by the user who enters their own token.

For a public deployment where you do not want surprise charges, leave `HF_TOKEN` unset and keep
`ALLOW_SESSION_TOKENS=true`. Visitors can still explore Sketch mode, and power users can bring
their own Hugging Face token for real output.

## Get a Hugging Face Token

You only need a token for real cloud generation. Sketch mode works without one.

1. Create or log into a Hugging Face account at `https://huggingface.co`.
2. Open `https://huggingface.co/settings/tokens`.
3. Click `New token` or `Create new token`.
4. Use a clear token name, such as `imageforge-local-test` or `imageforge-streamlit-prod`.
5. For local testing, choose `Read`. For production, Hugging Face recommends fine-grained tokens
   when you want tighter access control.
6. Copy the token once. It should look like `hf_...`.
7. Never paste the token into chat, commit it to git, or put it in README examples with the real value.

Official reference: [Hugging Face User Access Tokens](https://huggingface.co/docs/hub/security-tokens).

## Local Secrets

For local development, paste the token into `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_your_token_here"
IMAGEFORGE_PROFILE = "local"
ALLOW_LIVE_FALLBACK = "false"
ALLOW_SESSION_TOKENS = "true"
ALLOW_DEMO_MODE = "true"
```

The real `.streamlit/secrets.toml` file is ignored by git. Use
`.streamlit/secrets.example.toml` as the shareable template.

Streamlit also supports global secrets:

- macOS/Linux: `~/.streamlit/secrets.toml`
- Windows: `%userprofile%/.streamlit/secrets.toml`

Official reference: [Streamlit secrets.toml](https://docs.streamlit.io/develop/api-reference/connections/secrets.toml).

## Live Generation

After saving `HF_TOKEN`, restart Streamlit so the app reads the new secret:

```bash
streamlit run app.py
```

In the UI:

1. Confirm the sidebar says `Live provider is connected with a server-side secret`.
2. Turn off `Sketch mode`.
3. Click `Generate`.
4. Confirm the result source says `Hugging Face`.

You can also set a token for one terminal command without editing `secrets.toml`:

```bash
HF_TOKEN=hf_your_token_here streamlit run app.py
```

## Environment Profiles

Use these environment variables or Streamlit secrets to control behavior:

| Name | Local dev | Public no-cost demo | Owner-funded production |
| --- | --- | --- | --- |
| `IMAGEFORGE_PROFILE` | `local` | `production` | `production` |
| `HF_TOKEN` | optional | unset | set as secret |
| `ALLOW_LIVE_FALLBACK` | `false` | `false` | `false` |
| `ALLOW_SESSION_TOKENS` | `true` | `true` | `false` |
| `ALLOW_DEMO_MODE` | `true` | `true` | `true` |

Recommended public no-cost secrets:

```toml
IMAGEFORGE_PROFILE = "production"
ALLOW_LIVE_FALLBACK = "false"
ALLOW_SESSION_TOKENS = "true"
ALLOW_DEMO_MODE = "true"
```

Recommended owner-funded secrets:

```toml
HF_TOKEN = "hf_your_token_here"
IMAGEFORGE_PROFILE = "production"
ALLOW_LIVE_FALLBACK = "false"
ALLOW_SESSION_TOKENS = "false"
ALLOW_DEMO_MODE = "true"
```

## Testing Generation

Run a no-cost smoke test first. It writes a real PNG produced by Sketch mode:

```bash
.venv/bin/python scripts/smoke_generation.py
```

Then test live generation with fallback disabled. This proves the image came from Hugging Face:

```bash
.venv/bin/python scripts/smoke_generation.py --live --quality Balanced --output outputs/smoke-live.png
```

The smoke test reads `HF_TOKEN` from `.streamlit/secrets.toml`, the environment, or `--token`.
Generated smoke-test files are written into `outputs/`, which is ignored by git.

See [DEPLOYMENT.md](./DEPLOYMENT.md) for a practical no/low-cost deployment path.

## Production Deployment Summary

The simplest production path is Streamlit Community Cloud:

1. Push this repo to GitHub.
2. Go to `https://share.streamlit.io`.
3. Click `Create app`.
4. Select the repo, branch, and `app.py`.
5. In `Advanced settings`, paste the appropriate secrets block from `DEPLOYMENT.md`.
6. Deploy and test Sketch mode plus live mode if `HF_TOKEN` is configured.

Official references:

- [Deploy your app on Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit Community Cloud secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

Note: the default `Auto showcase` model is `FLUX.1-schnell`. The Hugging Face route currently
uses Together AI underneath, where FLUX Schnell supports steps but not `guidance_scale`; the app
therefore maps live requests to `4` / `6` / `12` steps for Fast / Balanced / Showcase and omits
guidance for that model.

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

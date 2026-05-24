# Deployment Guide

This app can be deployed as a low-cost product without running your own GPU. The recommended
first launch is Streamlit Community Cloud plus Hugging Face Inference Providers.

## Deployment Targets

| Target | Best for | Cost behavior | Notes |
| --- | --- | --- | --- |
| Streamlit Community Cloud | Fast public portfolio launch | App hosting is free; live image generation depends on token/provider usage | Recommended first production target |
| Hugging Face Spaces - Streamlit SDK | AI demo hosted inside HF | CPU Spaces can be free; live generation still uses provider quota | Only port `8501` is allowed for Streamlit Spaces |
| Hugging Face Spaces - Docker | More control over image/build | Can be free on CPU; GPU hardware is billed while running | Set the external app port correctly |
| Self-hosted VM | Custom domain and full control | You pay the VM and any provider usage | More ops work; use only if needed |

## Recommended No/Low-Cost Launch

Use this setup when you want a polished public app but do not want surprise image-generation bills.

1. Push the repository to GitHub.
2. Go to `https://share.streamlit.io`.
3. Click `Create app`.
4. Select the GitHub repo, branch, and `app.py` as the entrypoint.
5. In `Advanced settings`, choose a supported Python version and paste the app secrets.
6. Deploy the app.
7. Open the deployed URL and run through the launch checklist below.

Streamlit's deployment flow asks for the repository, branch, and entrypoint file. Its Advanced
settings screen is also where you paste secrets instead of committing `.streamlit/secrets.toml`.

Use these secrets when you want no-cost public browsing:

```toml
IMAGEFORGE_PROFILE = "production"
ALLOW_LIVE_FALLBACK = "false"
ALLOW_SESSION_TOKENS = "true"
ALLOW_DEMO_MODE = "true"
LIVE_SESSION_LIMIT = "4"
MAX_LIVE_IMAGES = "1"
MAX_SKETCH_IMAGES = "4"
```

Leave `HF_TOKEN` unset if you do not want to pay for public visitors. With no `HF_TOKEN`, visitors
can still use Sketch mode. Users who want real output can enter their own Hugging Face token for
that browser session.

## Owner-Funded Live Generation

Use this when you want visitors to click Generate and receive real images without bringing a key:

```toml
HF_TOKEN = "hf_your_token_here"
IMAGEFORGE_PROFILE = "production"
ALLOW_LIVE_FALLBACK = "false"
ALLOW_SESSION_TOKENS = "false"
ALLOW_DEMO_MODE = "true"
LIVE_SESSION_LIMIT = "4"
MAX_LIVE_IMAGES = "1"
MAX_SKETCH_IMAGES = "4"
```

Keep `ALLOW_LIVE_FALLBACK=false` in production so provider failures show as real failures instead
of silently returning local sketch images.

## Bring-Your-Own-Token Production

Use this when you want to show a polished product without paying for other people's generations:

```toml
IMAGEFORGE_PROFILE = "production"
ALLOW_LIVE_FALLBACK = "false"
ALLOW_SESSION_TOKENS = "true"
ALLOW_DEMO_MODE = "true"
LIVE_SESSION_LIMIT = "4"
MAX_LIVE_IMAGES = "1"
MAX_SKETCH_IMAGES = "4"
```

In this mode:

- Visitors can use Sketch mode for free.
- Visitors can paste their own Hugging Face token for real images.
- Your deployment does not include an owner-funded `HF_TOKEN`.

`LIVE_SESSION_LIMIT` is a per-browser-session guard. It is not billing software, but it helps keep
casual public usage under control. `MAX_LIVE_IMAGES=1` keeps each live click to one provider call,
while `MAX_SKETCH_IMAGES` can stay higher because Sketch mode runs locally.

## Hugging Face Token Setup

1. Open `https://huggingface.co/settings/tokens`.
2. Create one token per environment.
3. Suggested names:
   - `imageforge-local-test`
   - `imageforge-streamlit-prod`
   - `imageforge-hf-space`
4. Use `Read` for local testing.
5. Prefer fine-grained tokens for production when you want narrower access.
6. Store the token in Streamlit Cloud secrets, HF Space secrets, or `.streamlit/secrets.toml`.
7. Rotate the token if it is ever pasted into chat, committed, or shared accidentally.

## Local Production Check

For local development, paste your token into `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_your_token_here"
IMAGEFORGE_PROFILE = "production"
ALLOW_LIVE_FALLBACK = "false"
ALLOW_SESSION_TOKENS = "true"
ALLOW_DEMO_MODE = "true"
LIVE_SESSION_LIMIT = "12"
MAX_LIVE_IMAGES = "1"
MAX_SKETCH_IMAGES = "4"
```

The real local secrets file is ignored by git. Use `.streamlit/secrets.example.toml` as the shareable template.

Run the app locally with production-like behavior:

```bash
IMAGEFORGE_PROFILE=production \
ALLOW_LIVE_FALLBACK=false \
ALLOW_SESSION_TOKENS=true \
ALLOW_DEMO_MODE=true \
LIVE_SESSION_LIMIT=12 \
MAX_LIVE_IMAGES=1 \
streamlit run app.py
```

If port `8501` is already in use on macOS or Linux:

```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
kill <PID>
```

Before sharing the URL, run the generation smoke tests:

```bash
.venv/bin/python scripts/smoke_generation.py
.venv/bin/python scripts/smoke_generation.py --live --quality Balanced --output outputs/smoke-live.png
```

The first command proves the free Sketch path can write a PNG. The second command proves real
Hugging Face generation works without falling back to Sketch mode. The live smoke test reads
`HF_TOKEN` from `.streamlit/secrets.toml`, the environment, or `--token`.

## Environment Matrix

| Environment | Token location | Command / entrypoint | Expected result |
| --- | --- | --- | --- |
| macOS local | `.streamlit/secrets.toml` or shell env | `streamlit run app.py` | Local app at `http://localhost:8501` |
| Windows local | `.streamlit/secrets.toml` or shell env | `start.bat` or `streamlit run app.py` | Local app at `http://localhost:8501` |
| Linux local | `.streamlit/secrets.toml` or shell env | `streamlit run app.py` | Local app at `http://localhost:8501` |
| Streamlit Community Cloud | Advanced settings secrets | `app.py` | Public `*.streamlit.app` URL |
| Hugging Face Streamlit Space | Space secrets | Streamlit SDK, port `8501` | Public Space URL |
| Hugging Face Docker Space | Space secrets | Dockerfile with Streamlit on `0.0.0.0:8501` | Public Space URL |

## Platform Notes

- Streamlit Community Cloud is the easiest free host for this codebase.
- Hugging Face Spaces can host Streamlit. Streamlit Spaces only allow port `8501`.
- Docker Spaces need their public `app_port` to match the Streamlit port, usually `8501`.
- Hugging Face GPU Spaces can run real local diffusion, but GPU hardware is billed while running.
- Replicate or fal can be added later as provider adapters if you want more predictable per-image
  pricing or faster queues.

Official references:

- Streamlit Community Cloud deploy: `https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy`
- Streamlit Community Cloud secrets: `https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management`
- Hugging Face tokens: `https://huggingface.co/docs/hub/security-tokens`
- Hugging Face Streamlit Spaces: `https://huggingface.co/docs/hub/main/spaces-sdks-streamlit`

## Launch Checklist

- Confirm `IMAGEFORGE_PROFILE=production`.
- Confirm `ALLOW_LIVE_FALLBACK=false`.
- Confirm whether `HF_TOKEN` is intentionally set or unset.
- Confirm `LIVE_SESSION_LIMIT` and `MAX_LIVE_IMAGES` are conservative for public sharing.
- Generate one Sketch mode image.
- Generate one live image if a token is configured.
- Check the sidebar Launch guard before sharing the URL.
- If owner-funded live generation is enabled, watch provider usage and be ready to remove `HF_TOKEN`.

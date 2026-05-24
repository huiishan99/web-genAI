# Deployment Guide

This app can be deployed as a low-cost product without running your own GPU. The recommended
first launch is Streamlit Community Cloud plus Hugging Face Inference Providers.

## Recommended No/Low-Cost Launch

Use this setup when you want a polished public app but do not want surprise image-generation bills.

1. Push the repository to GitHub.
2. Create a Streamlit Community Cloud app from the repo with `app.py` as the entrypoint.
3. In the app secrets, set:

```toml
IMAGEFORGE_PROFILE = "production"
ALLOW_LIVE_FALLBACK = "false"
ALLOW_SESSION_TOKENS = "true"
ALLOW_DEMO_MODE = "true"
```

4. Leave `HF_TOKEN` unset if you do not want to pay for public visitors.
5. Add `HF_TOKEN` only when you are ready for the app owner to fund live generation.

With no `HF_TOKEN`, visitors can still use Sketch mode. Users who want real output can enter their
own Hugging Face token for that browser session.

## Owner-Funded Live Generation

Set this when you want visitors to click Generate and receive real images without bringing a key:

```toml
HF_TOKEN = "hf_your_token_here"
IMAGEFORGE_PROFILE = "production"
ALLOW_LIVE_FALLBACK = "false"
ALLOW_SESSION_TOKENS = "false"
ALLOW_DEMO_MODE = "true"
```

Keep `ALLOW_LIVE_FALLBACK=false` in production so provider failures show as real failures instead
of silently returning local sketch images.

## Local Production Check

For local development, paste your token into `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_your_token_here"
```

The real local secrets file is ignored by git. Use `.streamlit/secrets.example.toml` as the shareable template.

Run the app locally with production-like behavior:

```bash
IMAGEFORGE_PROFILE=production \
ALLOW_LIVE_FALLBACK=false \
ALLOW_SESSION_TOKENS=true \
ALLOW_DEMO_MODE=true \
streamlit run app.py
```

Before sharing the URL, run the generation smoke tests:

```bash
.venv/bin/python scripts/smoke_generation.py
HF_TOKEN=hf_your_token_here .venv/bin/python scripts/smoke_generation.py --live
```

The first command proves the free Sketch path can write a PNG. The second command proves real
Hugging Face generation works without falling back to Sketch mode.

## Platform Notes

- Streamlit Community Cloud is the easiest free host for this codebase.
- Hugging Face Spaces can host Streamlit through the Docker template; use the default Streamlit
  port `8501`.
- Hugging Face GPU Spaces can run real local diffusion, but GPU hardware is billed while running.
- Replicate or fal can be added later as provider adapters if you want more predictable per-image
  pricing or faster queues.

## Launch Checklist

- Confirm `IMAGEFORGE_PROFILE=production`.
- Confirm `ALLOW_LIVE_FALLBACK=false`.
- Confirm whether `HF_TOKEN` is intentionally set or unset.
- Generate one Sketch mode image.
- Generate one live image if a token is configured.
- Check the sidebar Launch guard before sharing the URL.

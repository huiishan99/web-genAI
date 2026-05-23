# Devlog

## 2026-05-24 00:08 JST - Recoverable Image Generation Demo

### Context

The hackathon version looked strong on paper, but the demo path was brittle: generation required a Hugging Face token, several UI controls were not wired into the backend, and default setup pulled heavy local AI dependencies that made first-run setup slow and fragile.

### Changes

- Reworked `app.py` around a resilient generation flow with demo mode, live Hugging Face generation, and fallback rendering.
- Added real handling for model selection, style prompt enhancement, negative prompt, seed offsets, batch size, quality presets, image size, denoising steps, and guidance scale.
- Replaced the token-blocking UX with a local deterministic renderer so the app can be presented without network access, GPU access, or provider credits.
- Added a small in-session gallery, download buttons, refreshed sidebar stats, and clearer recovery notes.
- Updated Hugging Face integration to prefer `huggingface_hub.InferenceClient.text_to_image`, with legacy HTTP fallback.
- Slimmed default dependencies to the API-first runtime and moved heavier hardware packages into optional extras.
- Updated launchers and docs to require Python 3.9+ and install the lighter dependency set.
- Added `.playwright-cli/` to `.gitignore` for local browser verification artifacts.

### Verification

- `env PYTHONPYCACHEPREFIX=/private/tmp/web-genai-pycache .venv/bin/python -m py_compile app.py launcher.py`
- `bash -n start.sh`
- `.venv/bin/python -m pip install -r requirements.txt`
- `.venv/bin/python -m pip install . --no-deps --force-reinstall`
- Browser verification with Playwright at `http://localhost:8501`
- Clicked `Generate` in demo mode and confirmed generated image, download button, and sidebar stats.

### Follow-Up

- Test live Hugging Face generation with a real token.
- Decide whether the procedural demo renderer should stay abstract or use bundled showcase images for stronger visual quality.
- Add automated tests around prompt enhancement, seed handling, and fallback behavior if this becomes more than a demo.

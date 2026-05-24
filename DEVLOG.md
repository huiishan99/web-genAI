# Devlog

## 2026-05-24 23:50 JST - Product Copy and README Polish

### Context

The README still read like a rescue report, and the app UI exposed a few implementation details that made the product feel less polished for first-time users.

### Changes

- Rewrote the README around a first-time user journey: what the app does, local setup, live generation, smoke tests, deployment modes, models, and project map.
- Removed the old `Why This Version Works Better` framing and shortened the README's internal implementation commentary.
- Refined the app's masthead, sidebar, prompt, empty state, and render status copy to feel more product-facing.
- Removed the in-app `Build notes` expander.
- Simplified the preset strip so it does not show live-incompatible internal steps/guidance values.
- Updated live success copy to avoid exposing provider parameter details in the main UI.
- Added a generator cache version so Streamlit does not reuse stale generation methods after UI copy changes.
- Refreshed the README screenshot to match the polished UI and live result copy.

### Verification

- `env PYTHONPYCACHEPREFIX=/private/tmp/web-genai-pycache .venv/bin/python -m py_compile app.py launcher.py scripts/smoke_generation.py`
- `.venv/bin/python scripts/smoke_generation.py`
- Browser verification at `http://localhost:8501`
- Refreshed `assets/readme-live-generation.png` from the updated UI.

## 2026-05-24 12:20 JST - README Deployment Guide Expansion

### Context

The project needed clearer handoff documentation for people who have not followed the rescue process. The README needed to show what the product looks like, explain how to obtain a Hugging Face token, and cover local testing on macOS, Windows, and Linux as well as production deployment options.

### Changes

- Added a real live-generation UI screenshot to `assets/readme-live-generation.png` and embedded it near the top of the README.
- Expanded README quick starts for macOS, Windows, and Linux.
- Added detailed Hugging Face token setup steps and local Streamlit secrets instructions.
- Added environment profile examples for local, public no-cost, and owner-funded production usage.
- Reworked `DEPLOYMENT.md` with deployment targets, Streamlit Community Cloud steps, Hugging Face Spaces notes, and a launch checklist.
- Added official documentation links for Hugging Face tokens, Streamlit secrets, Streamlit Community Cloud deployment, and HF Streamlit Spaces.

### Verification

- Captured the README screenshot from the live local Streamlit page at `http://localhost:8501`.
- Confirmed `assets/readme-live-generation.png` exists and is tracked as a committed asset.
- Reviewed README and deployment docs for local setup, token handling, and environment-specific flows.

## 2026-05-24 12:14 JST - Hugging Face FLUX Parameter Fix

### Context

Live Hugging Face generation failed for `black-forest-labs/FLUX.1-schnell` with provider-specific parameter errors. The app was sending the general Balanced preset with 28 steps plus guidance, then falling back to Sketch output, which made the failure easy to miss.

### Changes

- Added model-specific live parameter handling for `FLUX.1-schnell`.
- Mapped Fast, Balanced, and Showcase live requests for that model to 4, 6, and 12 steps.
- Omitted guidance for `FLUX.1-schnell` because the Hugging Face route currently uses Together AI, whose compatibility table does not support `guidance_scale` for FLUX Schnell.
- Added a generation success note when live parameters are adjusted for provider compatibility.
- Changed the default live fallback setting to off unless explicitly enabled.
- Updated the smoke test to read `HF_TOKEN` from `.streamlit/secrets.toml` as well as the environment.
- Documented the FLUX step cap in the README.

### Verification

- `env PYTHONPYCACHEPREFIX=/private/tmp/web-genai-pycache .venv/bin/python -m py_compile app.py launcher.py scripts/smoke_generation.py`
- `.venv/bin/python -c "from app import AIImageGenerator, GenerationSettings; ..."`
- `.venv/bin/python scripts/smoke_generation.py`
- `.venv/bin/python scripts/smoke_generation.py --live --quality Balanced --output outputs/smoke-live.png`

## 2026-05-24 12:02 JST - Local Secrets File Setup

### Context

Live Hugging Face testing needs a token, but the token should never be pasted into chat or committed to git. The project needed a safe local place for development secrets plus a shareable template for future setup.

### Changes

- Added `.streamlit/secrets.toml` as the local file where the user can paste `HF_TOKEN`.
- Added `.streamlit/secrets.toml` to `.gitignore` so real tokens stay out of commits.
- Added `.streamlit/secrets.example.toml` as a safe committed template.
- Updated README and deployment docs with the local secrets workflow.

### Verification

- Confirmed `.streamlit/secrets.toml` is ignored by git.
- Confirmed the committed template contains only empty placeholder values.

## 2026-05-24 11:49 JST - Generation Smoke Test Script

### Context

The project needed a repeatable way to prove that image generation works beyond clicking through the UI. The test also needed to distinguish real provider output from the free Sketch renderer so live generation is not accidentally validated by fallback behavior.

### Changes

- Added `scripts/smoke_generation.py` to generate one PNG without launching Streamlit.
- Made the smoke test default to free Sketch mode and write to ignored `outputs/`.
- Added `--live` mode that requires `HF_TOKEN` or `--token` and keeps live fallback disabled by default.
- Updated README and deployment docs with exact commands for Sketch and live generation testing.

### Verification

- `env PYTHONPYCACHEPREFIX=/private/tmp/web-genai-pycache .venv/bin/python -m py_compile app.py launcher.py scripts/smoke_generation.py`
- `.venv/bin/python scripts/smoke_generation.py`
- `.venv/bin/python scripts/smoke_generation.py --live`
- Confirmed the smoke test writes a PNG into `outputs/`.
- Confirmed live smoke testing fails fast when `HF_TOKEN` is missing instead of falling back to Sketch output.

## 2026-05-24 11:20 JST - Low-Cost Product Launch Mode

### Context

The app needed a more product-ready path without forcing the owner to pay for every public image generation. The previous behavior was good for hackathon recovery, but live failures could still fall back to local demo imagery, which is confusing for a public launch.

### Changes

- Renamed the free local renderer path to `Sketch mode` so it is clearly separate from live AI output.
- Added deployment configuration through environment variables or Streamlit secrets.
- Added support for server-side `HF_TOKEN` and optional session-only user tokens.
- Added production-oriented launch controls for `IMAGEFORGE_PROFILE`, `ALLOW_LIVE_FALLBACK`, `ALLOW_SESSION_TOKENS`, and `ALLOW_DEMO_MODE`.
- Disabled fake-success behavior when live fallback is off, returning clear live-provider errors instead.
- Added launch guard status in the sidebar and a budget status tile in the masthead.
- Tightened the mobile masthead title so the product name does not awkwardly split on narrow screens.
- Added `DEPLOYMENT.md` and updated the README with no/low-cost deployment modes.

### Verification

- `env PYTHONPYCACHEPREFIX=/private/tmp/web-genai-pycache .venv/bin/python -m py_compile app.py launcher.py`
- `env PYTHONPYCACHEPREFIX=/private/tmp/web-genai-pycache .venv/bin/python -c "from app import AIImageGenerator, GenerationSettings; ..."`
- Browser verification with the in-app browser at `http://localhost:8501`
- Confirmed Sketch mode generation still works.
- Confirmed the sidebar reports profile, token source, and live fallback status.

## 2026-05-24 10:56 JST - Prompt Starter State Fix

### Context

Clicking a prompt starter such as `Fantasy` raised a StreamlitAPIException because the app tried to modify `st.session_state.current_prompt` after the text area with the same widget key had already been instantiated in the current run.

### Changes

- Added a small `apply_prompt_example` callback to update the prompt safely before Streamlit rerenders the page.
- Rewired prompt starter buttons to use `on_click` and stable widget keys.
- Removed the direct same-run session-state assignment and manual rerun from the prompt starter loop.

### Verification

- `env PYTHONPYCACHEPREFIX=/private/tmp/web-genai-pycache .venv/bin/python -m py_compile app.py launcher.py`
- Browser verification with the in-app browser at `http://localhost:8501`
- Clicked prompt starter buttons and confirmed the prompt updates without the Streamlit session-state error.

## 2026-05-24 01:00 JST - Responsive Layout Repair

### Context

After the Forge Studio visual pass, shrinking the browser exposed a responsive layout bug. Around tablet/narrow desktop widths, Streamlit could collapse or overlay the sidebar while the main workbench still tried to keep a desktop-style layout, causing the composition to look broken.

### Changes

- Added responsive CSS for the main prompt/output workbench so it stacks into one column below 900px.
- Added a sidebar-aware medium-width rule for the case where Streamlit keeps the sidebar expanded while the viewport is narrow.
- Reduced the empty render bay height on narrow screens so it does not dominate the viewport.
- Added mobile-specific spacing, masthead padding, one-column status tiles, and stacked preset metadata.
- Increased mobile top padding so the masthead no longer clips against the fixed Streamlit header.

### Verification

- `env PYTHONPYCACHEPREFIX=/private/tmp/web-genai-pycache .venv/bin/python -m py_compile app.py launcher.py`
- Browser verification with Playwright at `http://localhost:8501`
- Checked 760px width after resizing down from desktop.
- Checked 390px mobile width at the top of the page.
- Confirmed no horizontal overflow at 390px.
- Clicked `Generate` and confirmed output image, download button, and sidebar stats still work.

## 2026-05-24 00:35 JST - Prompt Starter Button Contrast Fix

### Context

The prompt starter buttons in the Forge Studio UI became difficult to read because the global button styling allowed Streamlit's button state colors to produce dark backgrounds with dark text.

### Changes

- Added explicit light paper-style backgrounds for non-primary Streamlit buttons.
- Forced dark ink text on prompt starter and download buttons across normal, hover, focus, and active states.
- Kept the primary `Generate` button in the dark forge style with light text.

### Verification

- `env PYTHONPYCACHEPREFIX=/private/tmp/web-genai-pycache .venv/bin/python -m py_compile app.py launcher.py`
- Browser verification with Playwright at `http://localhost:8501`
- Captured a screenshot confirming the prompt starter buttons are readable.
- Clicked `Generate` and confirmed image output, download button, and sidebar stats still work.

## 2026-05-24 00:22 JST - Forge Studio UI Pass

### Context

The rescued Streamlit app worked, but the interface still felt like a default AI demo: generic controls, generic headings, and little connection to the "ImageForge" identity. The user asked for a design direction with stronger taste and less AI-template energy.

### Inspiration

- Duct Tape AI suggested a practical image-generation workspace organized around real creative use cases.
- Prompt gallery products suggested keeping prompt starters visible, but treating them like creative directions instead of tutorial content.
- Dark creative dashboard references suggested separating a persistent control console from a lighter image workbench.

### Changes

- Added a custom visual system in `app.py` with a paper grid background, dark forge-console sidebar, brass/coral/cyan accents, and stronger typography.
- Replaced the default Streamlit title area with a Forge Studio masthead that surfaces mode, model, quality, and canvas state.
- Reframed the main workspace as `Prompt bench` and `Render bay`, with a proofing-style empty state.
- Restyled text areas, buttons, metrics, generated images, preset metadata, and download controls.
- Shortened the `Architecture` prompt starter label to `Castle` to avoid awkward button wrapping.
- Adjusted sidebar form-field contrast after browser inspection.

### Verification

- `env PYTHONPYCACHEPREFIX=/private/tmp/web-genai-pycache .venv/bin/python -m py_compile app.py launcher.py`
- `.venv/bin/python -c "from app import AIImageGenerator, GenerationSettings; ..."`
- Browser verification with Playwright at `http://localhost:8501`
- Confirmed the first viewport renders the new Forge Studio layout.
- Clicked `Generate` in demo mode and confirmed image output, download button, and sidebar stats.

### Follow-Up

- Replace the procedural fallback image style with stronger branded showcase imagery if the demo renderer remains user-facing.
- Consider a full frontend framework later if Streamlit styling becomes too limiting.

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

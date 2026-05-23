# Devlog

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

"""
AI-ImageForge
=============
Streamlit app for text-to-image generation with graceful demo fallback.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

os.environ["PYTHONWARNINGS"] = "ignore"


STYLE_LABELS = {
    "realistic": "Realistic",
    "anime": "Anime",
    "artistic": "Artistic",
    "fantasy": "Fantasy",
    "cyberpunk": "Cyberpunk",
    "portrait": "Portrait",
}

STYLE_ENHANCERS = {
    "realistic": "photorealistic, highly detailed, natural light, professional photography",
    "anime": "anime illustration, clean line art, expressive lighting, high quality artwork",
    "artistic": "expressive digital painting, layered brushwork, gallery-grade composition",
    "fantasy": "fantasy art, magical atmosphere, cinematic lighting, intricate detail",
    "cyberpunk": "cyberpunk city, neon reflections, futuristic technology, dramatic contrast",
    "portrait": "portrait photography, detailed face, professional studio lighting",
}

QUALITY_PRESETS = {
    "Fast": {"steps": 18, "guidance": 6.0, "size": 512},
    "Balanced": {"steps": 28, "guidance": 7.5, "size": 768},
    "Showcase": {"steps": 38, "guidance": 8.5, "size": 1024},
}

MODEL_OPTIONS = {
    "Auto showcase": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Stable Diffusion v1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "OpenJourney": "prompthero/openjourney",
}

EXAMPLES = {
    "Fantasy": "A majestic red dragon flying over a mountain observatory at sunset",
    "Architecture": "Ancient castle on a floating island surrounded by waterfalls",
    "Anime": "A quiet anime train platform under cherry blossoms after rain",
    "Landscape": "Serene mountain lake at sunrise with perfect reflections",
    "Cyberpunk": "Futuristic street market with neon signs and hovering vehicles",
    "Portrait": "Elegant royal portrait with ornate crown and soft studio lighting",
}


@dataclass(frozen=True)
class GenerationSettings:
    style: str
    quality: str
    model: str
    negative_prompt: str
    seed: int
    batch_size: int
    demo_mode: bool

    @property
    def preset(self) -> Dict[str, float]:
        return QUALITY_PRESETS[self.quality]


@dataclass
class GenerationResult:
    image_data: Optional[bytes]
    message: str
    source: str
    model: str
    seed: int


class HardwareOptimizer:
    def __init__(self):
        self.gpu_type = self.detect_gpu()
        self.optimizations = self.build_optimizations()

    def detect_gpu(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                if any(tag in gpu_name for tag in ("nvidia", "geforce", "rtx", "gtx")):
                    return "nvidia"
                if any(tag in gpu_name for tag in ("amd", "radeon")):
                    return "amd"
        except Exception:
            pass

        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "DmlExecutionProvider" in providers:
                return "amd"
        except Exception:
            pass

        return "cpu"

    def build_optimizations(self) -> Dict[str, Dict[str, object]]:
        return {
            "nvidia": {
                "acceleration": "CUDA",
                "memory_optimization": True,
                "fp16": True,
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            },
            "amd": {
                "acceleration": "DirectML",
                "memory_optimization": True,
                "fp16": False,
                "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
            },
            "cpu": {
                "acceleration": "CPU",
                "memory_optimization": False,
                "fp16": False,
                "providers": ["CPUExecutionProvider"],
            },
        }

    def get_optimization_info(self) -> Dict[str, object]:
        return self.optimizations[self.gpu_type]


class AIImageGenerator:
    def __init__(self):
        self.hardware = HardwareOptimizer()

    def enhance_prompt(self, prompt: str, style: str) -> str:
        enhancer = STYLE_ENHANCERS.get(style, "high quality, detailed")
        return f"{prompt.strip()}, {enhancer}"

    def generate_batch(
        self,
        prompt: str,
        api_token: str,
        settings: GenerationSettings,
    ) -> List[GenerationResult]:
        results = []
        for index in range(settings.batch_size):
            seed = settings.seed + index
            results.append(self.generate_one(prompt, api_token, settings, seed))
        return results

    def generate_one(
        self,
        prompt: str,
        api_token: str,
        settings: GenerationSettings,
        seed: int,
    ) -> GenerationResult:
        if settings.demo_mode or not api_token:
            image_data, message = self.generate_demo_image(prompt, settings.style, seed)
            return GenerationResult(image_data, message, "Demo", "Procedural fallback", seed)

        enhanced_prompt = self.enhance_prompt(prompt, settings.style)
        image_data, message = self.generate_with_hf_client(
            enhanced_prompt,
            api_token,
            settings,
            seed,
        )
        if image_data:
            return GenerationResult(image_data, message, "Hugging Face", settings.model, seed)

        fallback_data, fallback_message = self.generate_demo_image(prompt, settings.style, seed)
        return GenerationResult(
            fallback_data,
            f"{message} Fallback used: {fallback_message}",
            "Fallback",
            "Procedural fallback",
            seed,
        )

    def generate_with_hf_client(
        self,
        prompt: str,
        api_token: str,
        settings: GenerationSettings,
        seed: int,
    ) -> Tuple[Optional[bytes], str]:
        size = int(settings.preset["size"])
        steps = int(settings.preset["steps"])
        guidance = float(settings.preset["guidance"])

        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(
                provider="auto",
                api_key=api_token,
                timeout=120,
            )
            image = client.text_to_image(
                prompt=prompt,
                negative_prompt=settings.negative_prompt or None,
                height=size,
                width=size,
                num_inference_steps=steps,
                guidance_scale=guidance,
                model=settings.model,
                seed=seed,
            )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue(), f"Generated with {settings.model}"
        except ImportError:
            return self.generate_with_legacy_http(prompt, api_token, settings, seed)
        except Exception as exc:
            return None, f"Live generation failed: {exc}"

    def generate_with_legacy_http(
        self,
        prompt: str,
        api_token: str,
        settings: GenerationSettings,
        seed: int,
    ) -> Tuple[Optional[bytes], str]:
        api_url = f"https://api-inference.huggingface.co/models/{settings.model}"
        headers = {"Authorization": f"Bearer {api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": settings.negative_prompt,
                "num_inference_steps": int(settings.preset["steps"]),
                "guidance_scale": float(settings.preset["guidance"]),
                "width": int(settings.preset["size"]),
                "height": int(settings.preset["size"]),
                "seed": seed,
            },
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            content_type = response.headers.get("content-type", "")
            if response.status_code == 200 and content_type.startswith("image/"):
                return response.content, f"Generated with {settings.model}"
            return None, f"Legacy API returned HTTP {response.status_code}: {response.text[:160]}"
        except requests.RequestException as exc:
            return None, f"Legacy API request failed: {exc}"

    def generate_demo_image(self, prompt: str, style: str, seed: int) -> Tuple[Optional[bytes], str]:
        try:
            random_source = random.Random(self.seed_from_prompt(prompt, style, seed))
            width, height = 768, 768
            img = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(img)

            self.paint_background(draw, width, height, style, random_source)
            self.paint_prompt_elements(draw, prompt, width, height, random_source)
            self.paint_caption(draw, prompt, style, seed, width, height)

            buffer = io.BytesIO()
            img.save(buffer, format="PNG", quality=95)
            return buffer.getvalue(), "Demo image generated locally"
        except Exception as exc:
            return None, f"Demo generation failed: {exc}"

    def seed_from_prompt(self, prompt: str, style: str, seed: int) -> int:
        digest = hashlib.sha256(f"{prompt}|{style}|{seed}".encode("utf-8")).hexdigest()
        return int(digest[:12], 16)

    def paint_background(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        style: str,
        random_source: random.Random,
    ) -> None:
        palettes = {
            "realistic": [(47, 90, 120), (240, 180, 115), (28, 42, 58)],
            "anime": [(238, 143, 170), (110, 180, 218), (55, 64, 110)],
            "artistic": [(245, 203, 92), (56, 142, 129), (153, 67, 92)],
            "fantasy": [(95, 71, 143), (65, 154, 132), (235, 184, 102)],
            "cyberpunk": [(10, 20, 45), (0, 207, 222), (236, 65, 138)],
            "portrait": [(76, 50, 74), (220, 162, 124), (34, 56, 77)],
        }
        colors = palettes.get(style, palettes["realistic"])

        for y in range(height):
            ratio = y / max(height - 1, 1)
            mid_blend = min(1.0, ratio * 1.4)
            r = int(colors[0][0] * (1 - mid_blend) + colors[1][0] * mid_blend)
            g = int(colors[0][1] * (1 - mid_blend) + colors[1][1] * mid_blend)
            b = int(colors[0][2] * (1 - mid_blend) + colors[1][2] * mid_blend)
            if ratio > 0.58:
                low_blend = (ratio - 0.58) / 0.42
                r = int(r * (1 - low_blend) + colors[2][0] * low_blend)
                g = int(g * (1 - low_blend) + colors[2][1] * low_blend)
                b = int(b * (1 - low_blend) + colors[2][2] * low_blend)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        for _ in range(90):
            x = random_source.randint(0, width)
            y = random_source.randint(0, height)
            radius = random_source.randint(2, 16)
            alpha_color = tuple(min(255, c + random_source.randint(15, 60)) for c in random_source.choice(colors))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=alpha_color)

    def paint_prompt_elements(
        self,
        draw: ImageDraw.ImageDraw,
        prompt: str,
        width: int,
        height: int,
        random_source: random.Random,
    ) -> None:
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ("mountain", "landscape", "lake", "sunrise")):
            self.paint_mountains(draw, width, height)
        if any(word in prompt_lower for word in ("city", "cyberpunk", "neon", "street")):
            self.paint_city(draw, width, height, random_source)
        if any(word in prompt_lower for word in ("dragon", "fantasy", "magical")):
            self.paint_dragon_mark(draw, width, height, random_source)
        if any(word in prompt_lower for word in ("portrait", "face", "royal", "person", "girl")):
            self.paint_portrait_mark(draw, width, height)

    def paint_mountains(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        ridges = [
            [(0, height), (width * 0.18, height * 0.48), (width * 0.38, height * 0.74), (width * 0.55, height * 0.40), (width, height * 0.74), (width, height)],
            [(0, height), (width * 0.24, height * 0.62), (width * 0.46, height * 0.50), (width * 0.72, height * 0.66), (width, height * 0.52), (width, height)],
        ]
        colors = [(38, 54, 70), (72, 82, 91)]
        for points, color in zip(ridges, colors):
            draw.polygon([(int(x), int(y)) for x, y in points], fill=color)

    def paint_city(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        random_source: random.Random,
    ) -> None:
        baseline = int(height * 0.78)
        x = 0
        while x < width:
            building_width = random_source.randint(34, 76)
            building_height = random_source.randint(120, 310)
            color = random_source.choice([(18, 28, 56), (28, 25, 64), (35, 38, 72)])
            draw.rectangle((x, baseline - building_height, x + building_width, baseline), fill=color)
            for wx in range(x + 8, x + building_width - 8, 18):
                for wy in range(baseline - building_height + 16, baseline - 8, 30):
                    if random_source.random() > 0.45:
                        draw.rectangle((wx, wy, wx + 7, wy + 12), fill=random_source.choice([(0, 210, 220), (240, 72, 150), (247, 207, 89)]))
            x += building_width + random_source.randint(4, 12)

    def paint_dragon_mark(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        random_source: random.Random,
    ) -> None:
        center_x = width // 2
        center_y = int(height * 0.44)
        wing = [(center_x - 190, center_y + 20), (center_x - 40, center_y - 88), (center_x - 10, center_y + 40)]
        other_wing = [(center_x + 190, center_y + 20), (center_x + 40, center_y - 88), (center_x + 10, center_y + 40)]
        draw.polygon(wing, fill=(74, 36, 59))
        draw.polygon(other_wing, fill=(74, 36, 59))
        draw.ellipse((center_x - 46, center_y - 28, center_x + 46, center_y + 72), fill=(118, 43, 51))
        for i in range(9):
            x = center_x - 80 + i * 20
            y = center_y + 78 + random_source.randint(-8, 8)
            draw.ellipse((x, y, x + 28, y + 20), fill=(102, 36, 46))

    def paint_portrait_mark(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        cx = width // 2
        face_top = int(height * 0.28)
        draw.ellipse((cx - 95, face_top, cx + 95, face_top + 210), fill=(221, 164, 127))
        draw.arc((cx - 120, face_top - 36, cx + 120, face_top + 210), 190, 350, fill=(65, 48, 62), width=28)
        draw.ellipse((cx - 44, face_top + 82, cx - 24, face_top + 102), fill=(35, 39, 51))
        draw.ellipse((cx + 24, face_top + 82, cx + 44, face_top + 102), fill=(35, 39, 51))
        draw.arc((cx - 34, face_top + 125, cx + 34, face_top + 165), 20, 160, fill=(122, 54, 72), width=4)

    def paint_caption(
        self,
        draw: ImageDraw.ImageDraw,
        prompt: str,
        style: str,
        seed: int,
        width: int,
        height: int,
    ) -> None:
        font = ImageFont.load_default()
        caption = prompt.strip()
        if len(caption) > 70:
            caption = f"{caption[:67]}..."
        overlay_top = height - 96
        draw.rectangle((0, overlay_top, width, height), fill=(0, 0, 0))
        draw.text((28, overlay_top + 20), "AI-ImageForge demo render", fill=(255, 255, 255), font=font)
        draw.text((28, overlay_top + 42), f"{STYLE_LABELS.get(style, style)} | seed {seed}", fill=(220, 220, 220), font=font)
        draw.text((28, overlay_top + 64), caption, fill=(200, 200, 200), font=font)


@st.cache_resource(show_spinner=False)
def get_generator() -> AIImageGenerator:
    return AIImageGenerator()


def initialize_state() -> None:
    defaults = {
        "generation_count": 0,
        "total_time": 0.0,
        "current_prompt": EXAMPLES["Fantasy"],
        "gallery": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_generation_stats(container) -> None:
    avg_time = st.session_state.total_time / max(st.session_state.generation_count, 1)
    with container.container():
        stat_cols = st.columns(2)
        stat_cols[0].metric("Generated", st.session_state.generation_count)
        stat_cols[1].metric("Avg time", f"{avg_time:.1f}s")


def render_sidebar(generator: AIImageGenerator):
    hardware_info = generator.hardware.get_optimization_info()

    with st.sidebar:
        st.header("System")
        metric_cols = st.columns(2)
        metric_cols[0].metric("Device", generator.hardware.gpu_type.upper())
        metric_cols[1].metric("Runtime", str(hardware_info["acceleration"]))

        with st.expander("Machine details"):
            st.write(f"OS: {platform.system()} {platform.release()}")
            st.write(f"Architecture: {platform.architecture()[0]}")
            st.write(f"Processor: {platform.processor() or 'Unknown'}")
            st.write(f"Memory optimization: {'Enabled' if hardware_info['memory_optimization'] else 'Disabled'}")
            st.write(f"FP16: {'Available' if hardware_info['fp16'] else 'Unavailable'}")

        st.header("Generation")
        api_token = st.text_input(
            "Hugging Face token",
            type="password",
            value=os.environ.get("HF_TOKEN", ""),
            help="Leave empty to use local demo mode.",
        )
        demo_mode = st.toggle("Demo mode", value=not bool(api_token))

        model_name = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
        style = st.selectbox("Style", list(STYLE_LABELS.keys()), format_func=lambda value: STYLE_LABELS[value])
        quality = st.select_slider("Quality", options=list(QUALITY_PRESETS.keys()), value="Balanced")
        batch_size = st.number_input("Images", min_value=1, max_value=4, value=1, step=1)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)

        negative_prompt = st.text_input(
            "Avoid",
            value="blurry, low quality, distorted, watermark, text artifacts",
        )

        if demo_mode:
            st.info("Demo mode renders locally, so the app is presentable without network or credits.")
        elif api_token:
            st.success("Live generation is ready.")
        else:
            st.warning("Add a token or switch on demo mode.")

        stats_container = st.empty()
        render_generation_stats(stats_container)

    settings = GenerationSettings(
        style=style,
        quality=quality,
        model=MODEL_OPTIONS[model_name],
        negative_prompt=negative_prompt.strip(),
        seed=int(seed),
        batch_size=int(batch_size),
        demo_mode=bool(demo_mode),
    )
    return api_token.strip(), settings, stats_container


def render_prompt_controls() -> str:
    st.subheader("Prompt")
    prompt = st.text_area(
        "Image description",
        key="current_prompt",
        height=140,
        label_visibility="collapsed",
        placeholder="Describe a scene, subject, mood, lighting, camera, and details.",
    )

    st.caption("Examples")
    cols = st.columns(3)
    for index, (label, example) in enumerate(EXAMPLES.items()):
        if cols[index % 3].button(label, width="stretch"):
            st.session_state.current_prompt = example
            st.rerun()

    return prompt


def render_result_card(result: GenerationResult, elapsed: float, index: int) -> None:
    if not result.image_data:
        st.error(result.message)
        return

    image = Image.open(io.BytesIO(result.image_data))
    st.image(image, width="stretch")
    st.caption(f"{result.source} | {result.model} | seed {result.seed} | {elapsed:.1f}s")

    file_name = f"ai_imageforge_{index}_{result.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    st.download_button(
        "Download",
        data=result.image_data,
        file_name=file_name,
        mime="image/png",
        width="stretch",
        key=f"download_{index}_{result.seed}_{len(result.image_data)}",
    )


def update_gallery(results: Iterable[GenerationResult]) -> None:
    successful = [result for result in results if result.image_data]
    st.session_state.gallery = (successful + st.session_state.gallery)[:8]


def main() -> None:
    st.set_page_config(
        page_title="AI-ImageForge",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()
    generator = get_generator()
    api_token, settings, stats_container = render_sidebar(generator)

    st.title("AI-ImageForge")
    st.markdown("A recoverable text-to-image demo with live Hugging Face generation and a dependable local fallback.")

    prompt_col, result_col = st.columns([0.95, 1.25], gap="large")
    with prompt_col:
        prompt = render_prompt_controls()
        generate = st.button("Generate", type="primary", width="stretch")

        preset = settings.preset
        st.write(
            f"Preset: {settings.quality} | {int(preset['size'])}x{int(preset['size'])} | "
            f"{int(preset['steps'])} steps | guidance {preset['guidance']}"
        )

    with result_col:
        st.subheader("Output")
        if generate:
            if not prompt.strip():
                st.warning("Add a prompt before generating.")
            else:
                start_time = time.time()
                with st.spinner("Generating image set..."):
                    results = generator.generate_batch(prompt, api_token, settings)
                elapsed = time.time() - start_time

                st.session_state.generation_count += len([result for result in results if result.image_data])
                st.session_state.total_time += elapsed
                update_gallery(results)
                render_generation_stats(stats_container)

                cols = st.columns(min(settings.batch_size, 2))
                for index, result in enumerate(results):
                    with cols[index % len(cols)]:
                        if result.source == "Fallback":
                            st.warning(result.message)
                        elif result.message:
                            st.success(result.message)
                        render_result_card(result, elapsed, index)
        elif st.session_state.gallery:
            cols = st.columns(2)
            for index, result in enumerate(st.session_state.gallery[:4]):
                with cols[index % 2]:
                    render_result_card(result, 0.0, index)
        else:
            st.info("Choose an example or write a prompt, then generate your first image.")

    with st.expander("Recovery notes"):
        st.write(
            "The app now supports a real demo path without a token, live generation through Hugging Face "
            "Inference Providers when a token is available, and reproducible batch outputs through seed offsets."
        )
        st.write(
            "If a provider or model is unavailable during a presentation, the local renderer keeps the workflow alive."
        )


if __name__ == "__main__":
    main()

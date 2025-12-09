"""
Generate synthetic observed trajectories (initial steps) for images using an LLM.

For each image in a folder, we send the image (resized/JPEG) and an optional caption
to the LLM and ask for plausible initial trajectories. The LLM reply is parsed and
saved as a text file in the Social-GAN format:
<frame_id> <ped_id> <x> <y>

Environment:
- Requires OPENAI_API_KEY.
- Assumes an OpenAI-compatible endpoint (configurable via --base-url).
"""

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import random

from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    import openai  # type: ignore


def encode_image(path: Path, max_edge: int, jpeg_quality: int) -> str:
    """Resize and JPEG-encode, then base64."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if max(img.size) > max_edge:
            img.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


def load_caption(image_path: Path, caption_suffix: str) -> str:
    cap_path = image_path.with_name(image_path.stem + caption_suffix)
    if cap_path.exists():
        return cap_path.read_text().strip()
    return ""


def call_llm(image_b64: str, caption: str, model: str, base_url: str, min_frames: int, max_frames: int, min_peds: int, coord_space: str) -> str:
    """Request synthetic trajectories; return raw text content."""
    units = "meters" if coord_space == "meter" else "pixels"
    prompt = (
        "You see a scene image. Using the scene and caption context, propose plausible "
        "observed pedestrian trajectories. Provide ONLY JSON with key 'trajectories'. "
        f"Each item: {{'ped_id': int, 'coords': list of [x, y] pairs in {units}}}. "
        f"For each pedestrian, choose a random length between {min_frames} and {max_frames} frames "
        "(frames ordered from 1 upward). "
        f"Include at least {min_peds} different pedestrians. "
        f"Example: {{\"trajectories\": [{{\"ped_id\":1,\"coords\":[[x1,y1],[x2,y2],...]}}]}}. "
        "No text outside JSON."
    )
    if caption:
        prompt += f"\nCaption: {caption}"

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]

    if OpenAI is not None:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.5,
        )
        return resp.choices[0].message.content or ""

    # Legacy fallback (no image support)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = base_url
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
                + "\n(Note: legacy client without image support; respond with generic plausible trajectories.)",
            }
        ],
        # temperature=0.5,
    )
    return resp["choices"][0]["message"]["content"]


def parse_json(text: str, min_frames: int, max_frames: int, min_peds: int) -> List[Tuple[int, List[List[float]]]]:
    def _try_load(s: str) -> Dict[str, Any]:
        return json.loads(s.strip())

    data: Dict[str, Any] = {}
    try:
        data = _try_load(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
        if match:
            data = _try_load(match.group(1))
        else:
            raise ValueError("LLM response not valid JSON.")

    trajs = []
    for item in data.get("trajectories", []):
        if not isinstance(item, dict):
            continue
        ped_id = item.get("ped_id", None)
        coords = item.get("coords", [])
        if ped_id is None or not isinstance(coords, Sequence):
            continue
        coords_list: List[List[float]] = []
        for c in coords:
            if (
                isinstance(c, Sequence)
                and len(c) == 2
                and all(isinstance(v, (int, float)) for v in c)
            ):
                coords_list.append([float(c[0]), float(c[1])])
        if len(coords_list) >= min_frames:
            coords_list = coords_list[:max_frames]
            trajs.append((int(ped_id), coords_list))
    return trajs


def fallback_trajs(min_peds: int, min_frames: int, max_frames: int, range_x: float, range_y: float, start_pid: int = 1) -> List[Tuple[int, List[List[float]]]]:
    """Generate simple straight-line trajectories in meters as fallback."""
    trajs: List[Tuple[int, List[List[float]]]] = []
    for i in range(min_peds):
        pid = start_pid + i
        x0 = (pid % 5) * (range_x / 5.0) + 0.5
        y0 = (pid // 5) * (range_y / 4.0) + 0.5
        dx = (random.random() - 0.5) * 0.4
        dy = (random.random() - 0.5) * 0.4
        frames = random.randint(min_frames, max_frames)
        coords = [[x0 + dx * t, y0 + dy * t] for t in range(frames)]
        trajs.append((pid, coords))
    return trajs


def write_traj_txt(trajs: List[Tuple[int, List[List[float]]]], out_path: Path) -> None:
    # Emit by frame, then ped_id: frame1 ped1, frame1 ped2, ...
    lines = []
    max_frames = max((len(coords) for _, coords in trajs), default=0)
    for frame_idx in range(1, max_frames + 1):
        for ped_id, coords in trajs:
            if frame_idx <= len(coords):
                x, y = coords[frame_idx - 1]
                lines.append(f"{int(frame_idx)}\t{int(ped_id)}\t{x:.4f}\t{y:.4f}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic observed trajectories for images using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to write trajectory txt files (default: input-dir).",
    )
    parser.add_argument(
        "--suffix",
        default="_bg.png",
        help="Only process files ending with this suffix.",
    )
    parser.add_argument(
        "--caption-suffix",
        default="_caption_chatgpt4.txt",
        help="Caption filename suffix to load alongside each image (optional).",
    )
    parser.add_argument(
        "--out-suffix",
        default=".txt",
        help="Suffix for output trajectory files (appended to stem).",
    )
    parser.add_argument("--min-frames", type=int, default=4, help="Minimum frames per pedestrian.")
    parser.add_argument("--max-frames", type=int, default=8, help="Maximum frames per pedestrian.")
    parser.add_argument(
        "--model",
        default="azure/gpt-4o",
        help="LLM model id.",
    )
    parser.add_argument(
        "--base-url",
        default="https://aikey-gateway.ivia.ch",
        help="Base URL for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=512,
        help="Resize longest edge to this many pixels before sending.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=70,
        help="JPEG quality for payload.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on images to process (<=0 means no cap).",
    )
    parser.add_argument(
        "--min-peds",
        type=int,
        default=6,
        help="Minimum number of pedestrians to request/ensure per scene.",
    )
    parser.add_argument(
        "--coord-space",
        choices=["pixel", "meter"],
        default="meter",
        help="Coordinate space to request from LLM.",
    )
    parser.add_argument(
        "--range-x",
        type=float,
        default=20.0,
        help="Approx scene width in meters (for fallback generation).",
    )
    parser.add_argument(
        "--range-y",
        type=float,
        default=20.0,
        help="Approx scene height in meters (for fallback generation).",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in args.input_dir.iterdir() if p.name.endswith(args.suffix)])
    if args.max_images is not None and args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise ValueError(f"No images matching suffix '{args.suffix}' in {args.input_dir}")

    for img_path in images:
        image_b64 = encode_image(img_path, args.max_edge, args.jpeg_quality)
        caption = load_caption(img_path, args.caption_suffix)
        raw = call_llm(image_b64, caption, args.model, args.base_url, args.min_frames, args.max_frames, args.min_peds, args.coord_space)
        try:
            trajs = parse_json(raw, args.min_frames, args.max_frames, args.min_peds)
        except Exception as e:
            print(f"[{img_path.name}] Failed to parse LLM JSON ({e}); using fallback.")
            trajs = []

        if len(trajs) < args.min_peds:
            missing = args.min_peds - len(trajs)
            print(f"[{img_path.name}] Not enough trajectories ({len(trajs)}); adding {missing} fallback.")
            max_pid = max([p for p, _ in trajs], default=0)
            fallback = fallback_trajs(missing, args.min_frames, args.max_frames, args.range_x, args.range_y, start_pid=max_pid + 1)
            trajs = trajs + fallback

        out_path = output_dir / (img_path.stem + args.out_suffix)
        write_traj_txt(trajs, out_path)
        print(f"[{img_path.name}] -> {out_path} ({len(trajs)} trajectories)")


if __name__ == "__main__":
    main()

"""
Quick-and-dirty script to ask an LLM for a fake homography and obstacle polygons,
then save an oracle mask and homography matrix.

Requirements:
- OPENAI_API_KEY set in env.
- Network access to the configured LLM endpoint.
- No OpenCV dependency; uses numpy + PIL only.
"""

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

import openai


def encode_image(image_path: Path, max_edge: int, jpeg_quality: int) -> str:
    """
    Resize and JPEG-encode to keep payload small, then base64 encode.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if max_edge and max(img.size) > max_edge:
            img.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


def call_llm(image_b64: str, model: str) -> str:
    """
    Call LLM to get a JSON with homography and polygons.
    Expected JSON:
    {"homography": [[...3 values...], ... 3 rows ...],
     "polygons": [[[x,y],[x,y],...], ...]}
    """
    client = openai.Client(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://aikey-gateway.ivia.ch",
    )

    prompt = (
        "Given the attached image, propose a plausible 3x3 homography matrix "
        "(image pixels to a top-down ground plane) and a few obstacle polygons "
        "in image pixel coordinates. Return JSON ONLY with keys "
        "\"homography\" and \"polygons\". "
        "Homography rows should each have 3 numbers. "
        "Polygons is a list of polygons; each polygon is a list of [x, y] points. "
        "If unsure, use identity homography and an empty polygon list."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def parse_llm_json(text: str) -> Tuple[np.ndarray, List[List[Tuple[float, float]]]]:
    def _try_load(candidate: str) -> Dict[str, Any]:
        candidate = candidate.strip()
        return json.loads(candidate)

    data: Dict[str, Any] = {}
    try:
        data = _try_load(text)
    except json.JSONDecodeError:
        # Try to extract from fenced code block
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
        if match:
            try:
                data = _try_load(match.group(1))
            except json.JSONDecodeError:
                raise ValueError("LLM response was not valid JSON.")
        else:
            raise ValueError("LLM response was not valid JSON.")

    H = np.eye(3, dtype=np.float64)
    if "homography" in data:
        mat = np.asarray(data["homography"], dtype=np.float64)
        if mat.shape == (3, 3):
            H = mat

    polygons: List[List[Tuple[float, float]]] = []
    polys = data.get("polygons", [])
    if isinstance(polys, Sequence):
        for poly in polys:
            if not isinstance(poly, Sequence):
                continue
            pts = []
            for pt in poly:
                if (
                    isinstance(pt, Sequence)
                    and len(pt) == 2
                    and all(isinstance(v, (int, float)) for v in pt)
                ):
                    pts.append((float(pt[0]), float(pt[1])))
            if len(pts) >= 3:
                polygons.append(pts)

    return H, polygons


def draw_oracle(size: Tuple[int, int], polygons: Sequence[Sequence[Tuple[float, float]]], out_path: Path) -> None:
    mask = Image.new("L", size, color=0)
    if polygons:
        drawer = ImageDraw.Draw(mask)
        for poly in polygons:
            drawer.polygon(poly, fill=255)
    mask.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fake homography and oracle mask using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--image", type=Path, help="Single input reference image.")
    src_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of images (png) to traverse sequentially (one request per image).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Prefix for outputs for single-image mode (creates <prefix>_H.txt and <prefix>_oracle.png).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory when traversing a folder; filenames derive from image stems.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on how many images to process when traversing a folder (still one request at a time). <=0 means no cap.",
    )
    parser.add_argument(
        "--model",
        default="azure/gpt-4o",
        help="LLM model id to use (passed to OpenAI client).",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=512,
        help="Resize longest image edge to this many pixels before sending (reduces payload).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=60,
        help="JPEG quality for payload (lower -> smaller payload).",
    )
    args = parser.parse_args()

    def process_image(image_path: Path, output_prefix: Path) -> None:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_b64 = encode_image(image_path, args.max_edge, args.jpeg_quality)
        raw_response = call_llm(image_b64, args.model)

        try:
            H, polygons = parse_llm_json(raw_response)
        except Exception as e:
            print(f"Failed to parse LLM JSON for {image_path} ({e}); using identity and empty mask.")
            H = np.eye(3, dtype=np.float64)
            polygons = []

        # Save homography
        out_H = output_prefix.with_name(output_prefix.name + "_H.txt")
        out_H.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_H, H, fmt="%.8f")

        # Save oracle mask
        with Image.open(image_path) as img:
            size = img.size
        out_oracle = output_prefix.with_name(output_prefix.name + "_oracle.png")
        draw_oracle(size, polygons, out_oracle)

        # Save raw response for debugging
        out_raw = output_prefix.with_name(output_prefix.name + "_llm_raw.txt")
        out_raw.write_text(raw_response)

        print(f"[{image_path.name}] Homography -> {out_H}")
        print(f"[{image_path.name}] Oracle mask -> {out_oracle}")
        print(f"[{image_path.name}] Raw LLM response -> {out_raw}")

    if args.image:
        if args.output_prefix is None:
            raise ValueError("--output-prefix is required when using --image.")
        process_image(args.image, args.output_prefix)
        return

    # Folder traversal mode (one request per image)
    if args.output_dir is None:
        raise ValueError("--output-dir is required when using --input-dir.")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in args.input_dir.iterdir() if p.suffix.lower() == ".png"])
    if args.max_images is not None and args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise ValueError(f"No .png files found in {args.input_dir}")

    for img_path in images:
        prefix = args.output_dir / img_path.stem
        process_image(img_path, prefix)


if __name__ == "__main__":
    main()

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
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

import openai


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
    try:
        data: Dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
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
    parser.add_argument("--image", required=True, type=Path, help="Input reference image.")
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Prefix for outputs (creates <prefix>_H.txt and <prefix>_oracle.png).",
    )
    parser.add_argument(
        "--model",
        default="azure/gpt-4o",
        help="LLM model id to use (passed to OpenAI client).",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image_b64 = encode_image(args.image)

    raw_response = call_llm(image_b64, args.model)

    try:
        H, polygons = parse_llm_json(raw_response)
    except Exception as e:
        print(f"Failed to parse LLM JSON ({e}); falling back to identity and empty mask.")
        H = np.eye(3, dtype=np.float64)
        polygons = []

    # Save homography
    out_H = args.output_prefix.with_name(args.output_prefix.name + "_H.txt")
    out_H.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_H, H, fmt="%.8f")

    # Save oracle mask
    with Image.open(args.image) as img:
        size = img.size
    out_oracle = args.output_prefix.with_name(args.output_prefix.name + "_oracle.png")
    draw_oracle(size, polygons, out_oracle)

    # Save raw response for debugging
    out_raw = args.output_prefix.with_name(args.output_prefix.name + "_llm_raw.txt")
    out_raw.write_text(raw_response)

    print(f"Homography saved to: {out_H}")
    print(f"Oracle mask saved to: {out_oracle}")
    print(f"Raw LLM response saved to: {out_raw}")


if __name__ == "__main__":
    main()

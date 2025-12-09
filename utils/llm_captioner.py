"""
Generate captions for images using an LLM.
- Traverses a directory and processes files matching a suffix (default: _bg.png).
- Sends each image (resized/JPEG) to the LLM and saves a caption text file.

Environment:
- Requires OPENAI_API_KEY.
- Uses the OpenAI client with a configurable base URL and model id.
"""

import argparse
import io
import os
from pathlib import Path
from typing import Optional

from PIL import Image
import base64

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


def call_llm_caption(
    image_b64: str,
    model: str,
    base_url: str,
) -> str:
    """Request a caption; return the raw text content."""
    prompt = (
        "You are an image captioner. "
        "Return a single concise caption (one sentence) for the attached image. "
        "Output only the caption text, no JSON, no explanations."
    )

    if OpenAI is not None:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()

    # Legacy openai 0.x fallback (no native image chat): send text-only notice
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = base_url
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
                + "\n(Note: legacy client without image support; respond with a generic caption.)",
            }
        ],
        temperature=0.4,
    )
    return resp["choices"][0]["message"]["content"].strip()


def caption_file(
    image_path: Path,
    output_path: Path,
    model: str,
    base_url: str,
    max_edge: int,
    jpeg_quality: int,
) -> None:
    image_b64 = encode_image(image_path, max_edge, jpeg_quality)
    caption = call_llm_caption(image_b64, model, base_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(caption)
    print(f"[{image_path.name}] -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caption images in a folder using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write captions (defaults to input-dir).",
    )
    parser.add_argument(
        "--suffix",
        default="_bg.png",
        help="Only process files ending with this suffix.",
    )
    parser.add_argument(
        "--out-suffix",
        default="_caption_llm.txt",
        help="Suffix for caption output files (appended to the stem before extension is removed).",
    )
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
        help="JPEG quality used for payload.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of images to process (<=0 means no cap).",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted([p for p in args.input_dir.iterdir() if p.name.endswith(args.suffix)])
    if args.max_images is not None and args.max_images > 0:
        candidates = candidates[: args.max_images]
    if not candidates:
        raise ValueError(f"No files matching suffix '{args.suffix}' in {args.input_dir}")

    for img_path in candidates:
        stem = img_path.stem.replace(args.suffix.replace(".png", ""), "").rstrip("_")
        out_name = img_path.stem + args.out_suffix
        out_path = output_dir / out_name
        caption_file(
            img_path,
            out_path,
            model=args.model,
            base_url=args.base_url,
            max_edge=args.max_edge,
            jpeg_quality=args.jpeg_quality,
        )


if __name__ == "__main__":
    main()

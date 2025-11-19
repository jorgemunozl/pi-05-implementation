"""
Utility helpers to run the PaliGemma multimodal model from Hugging Face Transformers.

Usage:
    poetry run python src/function_calling.py path/to/image.jpg "caption this image"
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

DEFAULT_MODEL_ID = "google/paligemma-3b-mix-224"


@dataclass
class PaliGemmaClient:
    """
    Lazily loads processor + model and exposes a call helper for reuse.
    """

    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 64
    temperature: float = 0.2
    _processor: Optional[AutoProcessor] = None
    _model: Optional[PaliGemmaForConditionalGeneration] = None

    @property
    def processor(self) -> AutoProcessor:
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._processor

    @property
    def model(self) -> PaliGemmaForConditionalGeneration:
        if self._model is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                model = model.to("cpu")
            self._model = model
        return self._model

    def __call__(self, image_path: Path, prompt: str) -> str:
        """
        Generates text conditioned on an image + prompt.
        """

        image = Image.open(image_path).convert("RGB")
        processor = self.processor
        model = self.model

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def call_paligemma(image_path: str | Path, prompt: str, model_id: str = DEFAULT_MODEL_ID) -> str:
    """
    Convenience function for single-shot usage.
    """

    client = PaliGemmaClient(model_id=model_id)
    return client(Path(image_path), prompt)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the PaliGemma Hugging Face model.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("prompt", type=str, help="Prompt or instruction for the model.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id to load (default: %(default)s).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0 for greedy).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    client = PaliGemmaClient(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    result = client(args.image, args.prompt)
    print(result)


if __name__ == "__main__":
    main()

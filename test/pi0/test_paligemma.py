"""
This paligemma model is designed to be fine tuned. It doesn't return
something, I tried with the it version, but it doesn't work either.
"""
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image

MODEL_ID = "google/paligemma-3b-pt-224"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    ).eval()

    image = load_image("2.jpg")

    prompt = "<image>\ncaption:"

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    print(processor.tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

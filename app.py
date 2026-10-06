import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)


model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda",
)


OCR_PROMPT = """You are an OCR system for an image-text recognition challenge.

Extract ONLY the characters that belong to the target object in the image.

IMPORTANT:
- For a license plate, return ONLY the characters printed on the license plate.
- Ignore province, city, state, or country names if they are outside the license plate.
- Keep Chinese characters that are part of the license plate.
- Keep letters and numbers that are part of the target object.
- For road signs, return ALL visible words and numbers printed on the sign.
- Preserve the reading order.
- For multiple lines, read from top to bottom.
- Read each line from left to right.
- Do not explain anything.
- Do not add labels.
- Do not guess missing characters.
- Return ONLY the final visible text.
"""


def run_ocr(image_path, prompt=OCR_PROMPT, max_new_tokens=64):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-image",
        required=True,
        help="Path to the input image"
    )

    args = parser.parse_args()

    input_image = Path(args.input_image)

    result = run_ocr(str(input_image))

    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{input_image.stem}_output.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"text": result},
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"OCR: {result}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
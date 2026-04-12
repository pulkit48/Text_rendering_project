import torch
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# -------------------------
# Load model
# -------------------------
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

# -------------------------
# Load dataset
# CSV columns: winner, loser, prompt(optional)
# -------------------------
df = pd.read_csv("your_dataset.csv")

# -------------------------
# Prompt builder
# -------------------------
def build_prompt(prompt_text, label_first, label_second):
    return f"""
You are given a text prompt and two images.

Prompt: {prompt_text}

The first image is Image {label_first}.
The second image is Image {label_second}.

Which image better satisfies the prompt?

Consider:
- semantic correctness
- completeness
- visual quality

Answer strictly with one letter: {label_first} or {label_second}.
"""

# -------------------------
# Judge function
# -------------------------
def judge_pair(img_a, img_b, prompt_text, label_first="A", label_second="B"):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_a},
                {"type": "image", "image": img_b},
                {"type": "text", "text": build_prompt(prompt_text, label_first, label_second)},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    return output_text

# -------------------------
# Evaluation loop
# -------------------------
correct = 0
total = 0
skipped = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        win_path = row["winner"]
        lose_path = row["loser"]
        prompt = row.get("prompt", "Describe the image")

        img_win = Image.open(win_path).convert("RGB")
        img_lose = Image.open(lose_path).convert("RGB")

        # -------------------------
        # RANDOMIZE ORDER (CRITICAL)
        # -------------------------
        if random.random() > 0.5:
            img_a, img_b = img_win, img_lose
            correct_label = "A"
        else:
            img_a, img_b = img_lose, img_win
            correct_label = "B"

        response = judge_pair(img_a, img_b, prompt)

        # -------------------------
        # Parse output robustly
        # -------------------------
        response = response.upper()

        if "A" in response:
            pred = "A"
        elif "B" in response:
            pred = "B"
        else:
            skipped += 1
            continue

        if pred == correct_label:
            correct += 1

        total += 1

    except Exception as e:
        skipped += 1
        continue

# -------------------------
# Final Results
# -------------------------
agreement = correct / total if total > 0 else 0

print(f"\nTotal evaluated: {total}")
print(f"Skipped: {skipped}")
print(f"Agreement: {agreement:.4f}")

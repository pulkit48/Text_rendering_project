import torch
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# -------------------------
# Load model
# -------------------------
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# -------------------------
# Load dataset
# -------------------------
# CSV format: winner_path, loser_path
df = pd.read_csv("your_dataset.csv")

# -------------------------
# Prompt template
# -------------------------
def build_prompt(prompt_text, label_first="A", label_second="B"):
    return f"""
You are given a prompt and two images.

Prompt: {prompt_text}

The first image is Image {label_first}.
The second image is Image {label_second}.

Which image better satisfies the prompt in terms of:
- semantic correctness
- completeness
- visual quality

Answer strictly with one letter: {label_first} or {label_second}.
"""

# -------------------------
# Inference function
# -------------------------
def judge_pair(image_a, image_b, prompt_text, label_first="A", label_second="B"):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(prompt_text, label_first, label_second)},
                {"type": "image", "image": image_a},
                {"type": "image", "image": image_b},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)

    response = processor.decode(output[0], skip_special_tokens=True).strip()

    return response

# -------------------------
# Evaluation loop
# -------------------------
total = 0
correct = 0
skipped = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        win_path = row["winner"]
        lose_path = row["loser"]
        prompt = row.get("prompt", "Describe the image")  # optional

        img_win = Image.open(win_path).convert("RGB")
        img_lose = Image.open(lose_path).convert("RGB")

        # -------------------------
        # Randomize order (VERY IMPORTANT)
        # -------------------------
        if random.random() > 0.5:
            img_a, img_b = img_win, img_lose
            correct_label = "A"
        else:
            img_a, img_b = img_lose, img_win
            correct_label = "B"

        response = judge_pair(img_a, img_b, prompt)

        # -------------------------
        # Parse output
        # -------------------------
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
# Results
# -------------------------
agreement = correct / total if total > 0 else 0

print(f"Total evaluated: {total}")
print(f"Skipped: {skipped}")
print(f"Agreement: {agreement:.4f}")

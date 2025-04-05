import torch
import numpy as np
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image
# Load the CLIP model only once

_clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(prompt,image_path):
    """Computes the CLIP score for a given image and prompt."""
    image= Image.open(image_path)  # Ensure image is RGB
    image = np.array([image])  # Convert to numpy array
    images_int = (image * 255).astype("uint8")  # Convert to uint8 format
    tensor_image = torch.from_numpy(images_int).permute(0, 3, 1, 2)  # NHWC -> NCHW
    score = _clip_score_fn(tensor_image, prompt).detach()
    return round(float(score), 4)

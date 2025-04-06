# import ImageReward as RM
import hpsv2
from PIL import Image
# model = RM.load("ImageReward-v1.0")

# def image_reward(prompt, image_path):
#     """Computes the image reward for a given image and prompt."""
#     image = Image.open(image_path)  # Ensure image is RGB
#     return model.score(prompt, image)

def hpsv2_score(prompt, image_path):
    """Computes the HPSv2 score for a given image and prompt."""
    image = Image.open(image_path)  # Ensure image is RGB
    result = hpsv2.score(image, prompt, hps_version="v2.1") 
    if isinstance(result, list):
        result = result[0]
    return result
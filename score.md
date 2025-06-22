Code for vqa score:
If getting any error related to clip just do-
```python
pip install clip
```

If getting any error  as "cannot import name 'SiglipImageProcessor' from 'transformers'" then just do-
```python
pip install diffusers==0.30
```
```python

import t2v_metrics

# Initialize the scoring model
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl')  # our recommended scoring model

# Define image and caption
image = "temp.png"  # an image path in string format
text = "someone talks on the phone angrily while another person sits happily"

# Compute the score
score = clip_flant5_score(images=[image], texts=[text])

# Print the result
print("Score:", score)

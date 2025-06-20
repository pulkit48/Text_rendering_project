from datasets import load_dataset
import pandas as pd
import os
import numpy as np
import cv2

dataset = load_dataset("data-is-better-together/open-image-preferences-v1-binarized")
df = pd.DataFrame(dataset['train'])
df = df[['prompt', 'chosen', 'rejected']]

df=df.head(10)

root_dir='dataset1'
os.makedirs(root_dir, exist_ok=True)
os.makedirs(f"{root_dir}/win", exist_ok=True)
os.makedirs(f"{root_dir}/lose", exist_ok=True)

for ind,row in df.iterrows():
    print(ind)
    nparr = np.frombuffer(row['chosen']['bytes'], np.uint8)
    win_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
    nparr = np.frombuffer(row['rejected']['bytes'], np.uint8)
    lose_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    win_image = cv2.resize(win_image, (512, 512))
    lose_image = cv2.resize(lose_image, (512, 512))
    cv2.imwrite(f"{root_dir}/win/{ind}.png", win_image)
    cv2.imwrite(f"{root_dir}/lose/{ind}.png", lose_image)
    df.at[ind, 'chosen'] = f"{root_dir}/win/{ind}.png"
    df.at[ind, 'rejected'] = f"{root_dir}/lose/{ind}.png"
df.to_csv(f"{root_dir}/dataset.csv", index=False)

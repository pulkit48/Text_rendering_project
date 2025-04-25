import pandas as pd
import random

df = pd.read_csv("df.csv")

white_elements = [
    "whiteboard",
    "white slate",
    "white poster",
    "white canvas",
    "white sheet",
    "white paper",
    "white wall",
    "white plaster wall",
    "white porcelain tile",
    "white ceiling",
    "white drywall",
    "white backdrop",
    "white hospital wall",
    "white lab table",
    "white marble",
    "white sky",
    "white ceramic plate",
    "white curtain",
    "white frosted glass",
    "white projection screen",
    "white eggshell wall",
    "white chalkboard",
    "white tile floor",
    "white sand",
    "white linen",
    "white towel",
    "white notebook page",
    "white studio background",
    "white snow",
    "white pillowcase",
    "white bedspread",
    "white blanket",
    "whiteboard surface",
    "white light panel",
    "white kitchen counter",
    "white foam board",
    "white construction paper",
    "white plastic surface",
    "white computer screen",
    "white napkin",
    "white ceramic mug"
]

for i in range(len(df)):
    keyword = random.choice(white_elements)
    df.at[i, 'prompt'] = df.at[i, 'prompt'].replace("An image with white background", f"An image with a {keyword}")

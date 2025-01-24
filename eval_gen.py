import pandas as pd
import numpy as np
import random
# df=pd.read_csv('final_dataset.csv')

with open('eval_prompt1.txt') as f:
    eval_prompt1=f.readlines()

eval_prompt1=[i.strip() for i in eval_prompt1]

df=pd.read_csv('final_dataset.csv')
prompt_list=df['prompt'].tolist()

final_eval_list=[]

for i in eval_prompt1:
    flag=0
    for j in prompt_list:
        if i in j:
            # print(i)
            # print(j)
            flag=1
            break
    if flag==0:
        final_eval_list.append(i)
        


phrases = [
    "A book with a cover featuring a text",
    "A poster with bold text",
    "A billboard with striking text",
    "A magazine page with colorful text",
    "A digital screen displaying animated text",
    "A greeting card with handwritten text",
    "A banner with large, impactful text",
    "A flyer with promotional text",
    "A signboard with glowing text",
    "A webpage header with stylish text",
    "A notebook cover with engraved text",
    "A t-shirt design with creative text",
    "A coffee mug with motivational text",
    "A bookmark with inspiring text",
    "A wall calendar featuring elegant text",
]

type1=f"An Image with white background with a text '{{text}}' written on it."
type2=f"An image with a text '{{text}}' written on it."
type3=f"{random.choice(phrases)} '{{text}}'."

for i in final_eval_list:
    print(i)
    type=random.choice([type1,type2,type3])
    print(type.format(text=i))
    # break

output_file = "final_eval_list.txt"

with open(output_file, "w") as f:
    for item in final_eval_list:
        f.write(f"{item}\n")

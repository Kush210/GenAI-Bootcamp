### update your transformers library 
### pip install git+https://github.com/huggingface/transformers.git


# Make all the necessary imports
import torch
from transformers import pipeline
from pprint import pprint

# Define the pipeline with the task you want to pursue the model you want to use and and model weights precision 
pipe = pipeline(
    "fill-mask",
    model="answerdotai/ModernBERT-base",
    torch_dtype=torch.bfloat16,
)

input_text = "He walked to the [MASK]."
results = pipe(input_text)

# Display the result. In our mask case it will display mulple answers ranked in descending order based on the score.
pprint(results)



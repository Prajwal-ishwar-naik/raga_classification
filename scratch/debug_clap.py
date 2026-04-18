from transformers import ClapModel, ClapProcessor
import torch

model_id = "laion/clap-htsat-fused"
processor = ClapProcessor.from_pretrained(model_id)
model = ClapModel.from_pretrained(model_id)

inputs = processor(text=["a test"], return_tensors="pt")
outputs = model.get_text_features(**inputs)
print(f"Output type: {type(outputs)}")
print(f"Output attributes: {dir(outputs)}")

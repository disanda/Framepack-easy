from transformers import SiglipImageProcessor, SiglipVisionModel
from PIL import Image
import torch

# Step 1: 加载图像
image = Image.open("00066.png").convert("RGB")

# Step 2: 预处理
processor = SiglipImageProcessor.from_pretrained("./hf_download/lllyasviel_flux_redux_bfl", subfolder='feature_extractor')
inputs = processor(images=image, return_tensors="pt")
inputs["pixel_values"] = inputs["pixel_values"].half()  

# Step 3: 特征提取
model = image_encoder = SiglipVisionModel.from_pretrained("./hf_download/lllyasviel_flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
with torch.no_grad():
    outputs = model(**inputs)
    image_embedding = outputs.pooler_output
    
print(outputs.keys())
print(inputs.keys(), image, inputs["pixel_values"].shape) # 转为 float16, 384x384
print(outputs.pooler_output.shape) 
print(outputs.last_hidden_state.shape)
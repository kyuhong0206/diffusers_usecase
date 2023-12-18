from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

IMG_URL = 'test.png'
MASK_URL = 'test_mask.png'
MODEL_PATH = 'diffusers_dir_path'
PROMPT = 'test prompt, detail'
OUTPUT_PATH = 'output.png'
TI_PATH = 'test_ti.safetensors'

img_url = IMG_URL
mask_url = MASK_URL
model_path = MODEL_PATH
prompt = PROMPT

pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16, safety_checker=None, use_safetensors=True).to("cuda")
pipe.load_textual_inversion(TI_PATH)
init_image = Image.open(img_url).convert("RGB")
mask_image = Image.open(mask_url).convert("RGB")
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, strength=1, guidance_scale=9,width=1536, height=2048).images[0]

image.save(OUTPUT_PATH)




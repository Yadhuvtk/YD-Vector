import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import timm
from PIL import Image
import re

# --- SETTINGS ---
MODEL_ID = "./YD_Vector_Base"
CHECKPOINT = "./YD_Vector_Checkpoints/checkpoint-34500"
IMAGE_PATH = "1.jpg" # Put your test image here!

def clean_svg(text):
    # Remove the Byte-Tokens (Ġ, č, Ċ)
    clean = text.replace('Ġ', ' ').replace('č', '').replace('Ċ', '\n')
    
    # Simple regex to find the SVG tag and path
    svg_match = re.search(r'<svg.*?</svg>', clean, re.DOTALL)
    if svg_match:
        return svg_match.group(0)
    
    # Fallback: If it only output the path data, wrap it manually
    return f'<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="{clean}" fill="none" stroke="red" stroke-width="0.3"/></svg>'

# 1. Load Model (Optimized for 5090)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base, CHECKPOINT)

# 2. Prepare Vision Tower (SigLIP)
vision_tower = timm.create_model('vit_so400m_patch14_siglip_384', pretrained=True, num_classes=0).to("cuda").to(torch.bfloat16).eval()

# 3. Process Image
image = Image.open(IMAGE_PATH).convert("RGB").resize((384, 384))
# (Note: Ensure your specific projector logic is included here as per your training setup)

print(f"Vectorizing {IMAGE_PATH}...")
with torch.no_grad():
    # ... (Insert your specific image embedding logic here) ...
    
    output_ids = model.generate(
        inputs_embeds=image_embeddings,
        max_new_tokens=1024,
        temperature=0.1, # Keep it low for math stability
        repetition_penalty=2.0
    )
    
    raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    svg_result = clean_svg(raw_text)

with open("latest_output.svg", "w") as f:
    f.write(svg_result)
print("Done! Saved to latest_output.svg")
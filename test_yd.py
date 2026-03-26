import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import timm
from PIL import Image
import os
import re

# --- CONFIGURATION ---
BASE_MODEL_PATH = "./YD_Vector_Base"
CHECKPOINT_PATH = "./YD_Vector_Checkpoints/checkpoint-34500"
PROJECTOR_PATH = "yd_projector.bin" 
TEST_IMAGE = "1.jpg"

class YDProjector(nn.Module):
    def __init__(self, vision_dim=1152, model_dim=4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
    def forward(self, x): return self.projector(x)

def clean_svg_output(raw_text):
    # 1. Clean Byte-Tokens (Ġ=space, č=formatting, Ċ=newline, ĉ=extra space)
    text = raw_text.replace('Ġ', ' ').replace('č', '').replace('Ċ', '\n').replace('ĉ', ' ')
    
    # 2. Extract full <svg> if the model provided the tags
    svg_match = re.search(r'<svg.*?</svg>', text, re.DOTALL)
    if svg_match:
        return svg_match.group(0)

    # 3. FALLBACK: Advanced Filter
    # Includes 'a' for Arcs and 'm' for Relative moves learned at 33k+
    valid_parts = re.findall(r'[MLCVHZAmslcvhza]|-?[\d\.]+', text)
    cleaned_path = " ".join(valid_parts)
    
    # Wraps in a 512x512 viewBox to match your latest progress
    return f'''<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
  <path d="{cleaned_path}" fill="none" stroke="red" stroke-width="1.5" stroke-linecap="round" />
</svg>'''

def run_inference():
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: {TEST_IMAGE} not found.")
        return

    # Use Bfloat16 for RTX 5090 stability
    dtype = torch.bfloat16 

    print(f"--- Initializing YD-Vector Engine (34.5k) ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # Load Llama-3 (Removed Flash Attention 2 to fix Windows crash)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        quantization_config=bnb_config,
        device_map="auto",
        dtype=dtype # Replaces deprecated torch_dtype
    )
    
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()

    # Setup SigLIP Vision Tower
    print("Loading Vision Tower...")
    vision_tower = timm.create_model('vit_so400m_patch14_siglip_384', pretrained=True, num_classes=0).to("cuda").to(dtype).eval()
    
    # Proper Image Preprocessing for SigLIP
    data_config = timm.data.resolve_model_data_config(vision_tower)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Load Projector
    projector = YDProjector().to("cuda").to(dtype)
    if os.path.exists(PROJECTOR_PATH):
        projector.load_state_dict(torch.load(PROJECTOR_PATH, map_location="cuda"))
    projector.eval()

    # Process Input Image
    raw_image = Image.open(TEST_IMAGE).convert("RGB")
    img_tensor = transforms(raw_image).unsqueeze(0).to("cuda").to(dtype)

    print(f"Analyzing {TEST_IMAGE} and generating vector...")
    with torch.no_grad():
        image_features = vision_tower.forward_features(img_tensor)
        image_embeddings = projector(image_features).to(dtype)
        
        output_ids = model.generate(
            inputs_embeds=image_embeddings,
            max_new_tokens=2048,      # Increased for high-res 512 coordinates
            do_sample=True,
            temperature=0.2,          # Keep it low for mathematical precision
            repetition_penalty=3.0,   # High penalty to break the "Staircase" loop
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
        
        raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        final_svg = clean_svg_output(raw_output)

    print("\n--- SVG RESULT ---")
    print(final_svg)
    
    output_filename = "yd_34500_result.svg"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_svg)
    print(f"\nSuccess! Output saved to {output_filename}")

if __name__ == "__main__":
    run_inference()
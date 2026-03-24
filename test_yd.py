import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import timm
from PIL import Image
import os
import re

BASE_MODEL_PATH = "./YD_Vector_Base"
CHECKPOINT_PATH = "./YD_Vector_Checkpoints/checkpoint-16000"
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
    # 1. Clean up the byte tokens
    text = raw_text.replace('Ġ', ' ').replace('č', '').replace('Ċ', '\n').replace('ĉ', ' ')
    
    # 2. THE FILTER: Keep only SVG commands (M, L, C, H, V, Z) and numbers/decimals
    # This automatically deletes "github", "aria", "{ }", etc.
    valid_parts = re.findall(r'[MLCVHZmlcvhz]|-?[\d\.]+', text)
    cleaned_path = " ".join(valid_parts)
    
    # 3. Create a clean, standard SVG container
    # We add a red stroke so you can see the 'skeleton' of what it's drawing
    valid_svg = f'''<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path d="{cleaned_path}" fill="none" stroke="red" stroke-width="0.5" />
</svg>'''
    
    return valid_svg

def run_inference():
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: {TEST_IMAGE} not found.")
        return

    dtype = torch.bfloat16 

    print("--- Initializing YD-Vector Engine (16k) ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=dtype
    )
    
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()

    vision_tower = timm.create_model('vit_so400m_patch14_siglip_384', pretrained=True, num_classes=0).to("cuda").to(dtype).eval()
    projector = YDProjector().to("cuda").to(dtype)
    
    if os.path.exists(PROJECTOR_PATH):
        projector.load_state_dict(torch.load(PROJECTOR_PATH, map_location="cuda"))
    projector.eval()

    raw_image = Image.open(TEST_IMAGE).convert("RGB").resize((384, 384))
    img_tensor = torch.tensor(list(raw_image.getdata())).view(384, 384, 3).permute(2, 0, 1).unsqueeze(0).float().to("cuda").to(dtype) / 255.0

    print(f"Analyzing {TEST_IMAGE}...")
    with torch.no_grad():
        image_features = vision_tower.forward_features(img_tensor)
        image_embeddings = projector(image_features).to(dtype)
        
        output_ids = model.generate(
            inputs_embeds=image_embeddings,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            repetition_penalty=1.5,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
        
        raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        final_svg = clean_svg_output(raw_output)

    print("\n--- SVG RESULT ---")
    print(final_svg)
    
    with open("yd_16k_corrected.svg", "w", encoding="utf-8") as f:
        f.write(final_svg)
    print("\nOutput saved to yd_16k_corrected.svg")

if __name__ == "__main__":
    run_inference()
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, BitsAndBytesConfig
from peft import PeftModel
import timm
from PIL import Image
import os

# --- CONFIG ---
BASE_MODEL_PATH = "./YD_Vector_Base"
CHECKPOINT_PATH = "./YD_Vector_Checkpoints/checkpoint-2500"
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

def run_inference():
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: {TEST_IMAGE} not found.")
        return

    # Use BFloat16 - the native language of your RTX 5090
    dtype = torch.bfloat16 

    print("--- Loading YD-Vector Engine ---")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(BASE_MODEL_PATH)
    
    # Load 8B Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=dtype # Forces base layers to BF16
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    
    # PRECISION LOCK: Ensure the output head matches the rest of the model
    model.base_model.model.lm_head.to(dtype)
    model.eval()

    # Load Vision & Projector
    vision_tower = timm.create_model('vit_so400m_patch14_siglip_384', pretrained=True, num_classes=0).to("cuda").to(dtype).eval()
    projector = YDProjector().to("cuda").to(dtype)
    if os.path.exists(PROJECTOR_PATH):
        projector.load_state_dict(torch.load(PROJECTOR_PATH, map_location="cuda"))
    projector.eval()

    # Process Image
    raw_image = Image.open(TEST_IMAGE).convert("RGB").resize((384, 384))
    img_tensor = torch.tensor(list(raw_image.getdata())).view(384, 384, 3).permute(2, 0, 1).unsqueeze(0).float().to("cuda").to(dtype) / 255.0

    print(f"Tracing '{TEST_IMAGE}'...")
    with torch.no_grad():
        # Get visual features
        image_features = vision_tower.forward_features(img_tensor)
        
        # Ensure projector input is correct type
        image_embeddings = projector(image_features).to(dtype)
        
        # Generate tokens
        output_ids = model.generate(
            inputs_embeds=image_embeddings,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
        
        svg_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n--- GENERATED SVG ---")
    print(svg_output)
    
    with open("yd_test_output.svg", "w", encoding="utf-8") as f:
        f.write(svg_output)
    print("\nFile saved: yd_test_output.svg")

if __name__ == "__main__":
    run_inference()
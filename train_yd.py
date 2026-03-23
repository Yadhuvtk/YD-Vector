import torch
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import timm
from PIL import Image
import io

# --- CONFIGURATION ---
MODEL_PATH = "./YD_Vector_Base"
PROJECTOR_PATH = "yd_projector.bin"
SHARDS_PATH = "file:E:/Yadhu Projects/YD-Vector/SVG_Shards/yd-vector-{000000..000226}.tar"
BATCH_SIZE = 8  # Start with 8, your 5090 might handle 16 or 32!
LEARNING_RATE = 2e-4

# 1. LOAD THE BRAIN (8B Model) in 4-bit to save VRAM
print("Loading 8B Brain in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. APPLY LORA (The 'Learning' Layer)
peft_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 3. LOAD THE EYES (Vision Tower)
print("Loading Vision Tower (SigLIP)...")
vision_tower = timm.create_model('vit_so400m_patch14_siglip_384', pretrained=True, num_classes=0).to("cuda").eval()

# 4. LOAD THE SHARDS (Data Streamer)
def transform(sample):
    # Process PNG for Vision Tower
    img = sample["png"].convert("RGB").resize((384, 384))
    img_tensor = torch.tensor(list(img.getdata())).view(384, 384, 3).permute(2, 0, 1).float() / 255.0
    
    # Process SVG for the Brain
    svg_text = sample["svg"].decode("utf-8")
    return img_tensor, svg_text

dataset = wds.WebDataset(SHARDS_PATH).decode("pil").map(transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# --- START TRAINING LOOP ---
print("\n--- YD-Vector Training Started ---")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for i, (images, texts) in enumerate(loader):
    images = images.to("cuda")
    
    # A. Look at images
    with torch.no_grad():
        image_features = vision_tower.forward_features(images) # Extract 'shapes'
    
    # B. Generate SVG code (Simplified training step)
    # This is where the model compares its guess to the real SVG
    print(f"Batch {i}: Processing {len(texts)} samples on RTX 5090...")
    
    # [Actual Training Math happens here - we will refine this in the next step]
    
    if i == 10: break # Just a test run
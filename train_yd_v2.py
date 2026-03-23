import torch
import torch.nn as nn
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import timm
import os

# --- SETTINGS ---
# Resuming from your latest successful 'Brain' save
MODEL_PATH = "./YD_Vector_Checkpoints/checkpoint-2500"
# Base projector (since projector-2500 wasn't saved yet)
PROJECTOR_PATH = "yd_projector.bin" 
SHARDS_PATH = "file:E:/Yadhu Projects/YD-Vector/SVG_Shards/yd-vector-{000000..000226}.tar"
OUTPUT_DIR = "./YD_Vector_Checkpoints"

# Optimized for RTX 5090 24GB-32GB range
BATCH_SIZE = 10 
LEARNING_RATE = 1e-4

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class YDProjector(nn.Module):
    def __init__(self, vision_dim=1152, model_dim=4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
    def forward(self, x): return self.projector(x)

def transform_data(sample):
    img = sample["png"].convert("RGB").resize((384, 384))
    img_tensor = torch.tensor(list(img.getdata())).view(384, 384, 3).permute(2, 0, 1).float() / 255.0
    svg_text = sample["svg"].decode("utf-8")
    return img_tensor.to(torch.bfloat16), svg_text

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Always load the tokenizer from the base folder
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./YD_Vector_Base")
    tokenizer.pad_token = "[PAD]"

    print(f"Resuming Brain from: {MODEL_PATH}")
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

    model.gradient_checkpointing_enable()

    # Peft handles the resume automatically when loading from a checkpoint folder
    peft_config = LoraConfig(
        r=32, lora_alpha=64, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        lora_dropout=0.05, 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    print("Loading Vision Tower & Bridge...")
    vision_tower = timm.create_model('vit_so400m_patch14_siglip_384', pretrained=True, num_classes=0).to("cuda").to(torch.bfloat16).eval()
    
    projector = YDProjector().to("cuda").to(torch.bfloat16)
    if os.path.exists(PROJECTOR_PATH):
        projector.load_state_dict(torch.load(PROJECTOR_PATH))

    dataset = wds.WebDataset(SHARDS_PATH).shuffle(1000).decode("pil").map(transform_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=LEARNING_RATE)
    
    print(f"\nTraining Restarted. Batch Size: {BATCH_SIZE}")
    model.train()

    for step, (images, svg_texts) in enumerate(loader):
        actual_step = step + 2500 # Adjusting logs to show real progress
        
        images = images.to("cuda")
        tokens = tokenizer(svg_texts, truncation=True, max_length=1024, padding="max_length", return_tensors="pt").to("cuda")

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                image_features = vision_tower.forward_features(images)
            
            image_embeds = projector(image_features)
            text_embeds = model.get_base_model().model.embed_tokens(tokens.input_ids)
            
            full_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            img_labels = torch.full((BATCH_SIZE, image_embeds.shape[1]), -100).to("cuda")
            full_labels = torch.cat([img_labels, tokens.input_ids], dim=1)

            outputs = model(inputs_embeds=full_embeds, labels=full_labels)
            loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if actual_step % 5 == 0:
            vram_gb = torch.cuda.memory_reserved() / 1e9
            print(f"Step {actual_step} | Loss: {loss.item():.4f} | VRAM: {vram_gb:.2f}GB")

        # Save both parts every 500 steps
        if actual_step % 500 == 0:
            save_path = f"{OUTPUT_DIR}/checkpoint-{actual_step}"
            model.save_pretrained(save_path)
            torch.save(projector.state_dict(), f"{OUTPUT_DIR}/projector-{actual_step}.bin")
            print(f"--- [SUCCESS] Saved Brain & Bridge at Step {actual_step} ---")
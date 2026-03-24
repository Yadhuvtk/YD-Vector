import torch
import torch.nn as nn
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
import timm
import os
import time
import subprocess

BASE_MODEL = "./YD_Vector_Base"
CHECKPOINT_DIR = "./YD_Vector_Checkpoints/checkpoint-19500"
PROJECTOR_RESUME = "yd_projector.bin" 
OUTPUT_DIR = "./YD_Vector_Checkpoints"
SHARDS_PATH = "file:E:/Yadhu Projects/YD-Vector/SVG_Shards/yd-vector-{000000..000226}.tar"
BATCH_SIZE = 10 
LEARNING_RATE = 1e-4

# ===== COOLDOWN SETTINGS =====
COOLDOWN_AFTER_CHECKPOINT = 1     # Minutes to rest after each checkpoint save
# ==============================

dtype = torch.bfloat16 
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def get_gpu_temp():
    """Read GPU temperature via nvidia-smi. Returns temp in °C or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return None


def post_checkpoint_break(step, minutes=COOLDOWN_AFTER_CHECKPOINT):
    """Take a timed break after checkpoint save to let GPU rest."""
    total_seconds = minutes * 60
    
    print(f"\n{'='*60}")
    print(f"  GPU REST — Checkpoint {step} saved")
    print(f"  Taking a {minutes} minute break...")
    print(f"{'='*60}")
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        remaining = total_seconds - elapsed
        
        if remaining <= 0:
            break
        
        temp = get_gpu_temp()
        temp_str = f" | GPU: {temp}°C" if temp else ""
        mins_left = remaining / 60
        print(f"  Resuming in {mins_left:.1f} min{temp_str}   ", end="\r")
        time.sleep(5)
    
    temp = get_gpu_temp()
    temp_str = f" (GPU: {temp}°C)" if temp else ""
    print(f"\n  Break over{temp_str} — auto-resuming training!")
    print(f"{'='*60}\n")


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
    return img_tensor.to(dtype), svg_text


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # AUTO-CONFIG FIX
    config_path = os.path.join(CHECKPOINT_DIR, "adapter_config.json")
    if not os.path.exists(config_path):
        import json
        config_data = {
            "base_model_name_or_path": BASE_MODEL,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "bias": "none"
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = "[PAD]"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4"
    )

    print(f"Loading Base Brain...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=dtype
    )

    print(f"Resuming from Step 19500...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR, is_trainable=True)
    
    # PRECISION LOCK
    model.base_model.model.lm_head.to(dtype)
    model.gradient_checkpointing_enable()

    vision_tower = timm.create_model('vit_so400m_patch14_siglip_384', pretrained=True, num_classes=0).to("cuda").to(dtype).eval()
    projector = YDProjector().to("cuda").to(dtype)
    
    if os.path.exists(PROJECTOR_RESUME):
        projector.load_state_dict(torch.load(PROJECTOR_RESUME, map_location="cuda"))

    dataset = wds.WebDataset(SHARDS_PATH).shuffle(1000).decode("pil").map(transform_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=LEARNING_RATE)
    
    print(f"\nTraining Restarted. Current Step: 19500")
    print(f"GPU Rest: {COOLDOWN_AFTER_CHECKPOINT} min break after each checkpoint\n")
    model.train()

    for step, (images, svg_texts) in enumerate(loader):
        actual_step = step + 19500
        
        images = images.to("cuda").to(dtype)
        tokens = tokenizer(svg_texts, truncation=True, max_length=1024, padding="max_length", return_tensors="pt").to("cuda")

        with torch.amp.autocast('cuda', dtype=dtype):
            with torch.no_grad():
                image_features = vision_tower.forward_features(images)
            
            image_embeds = projector(image_features).to(dtype)
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
            gpu_temp = get_gpu_temp()
            temp_str = f" | GPU: {gpu_temp}°C" if gpu_temp else ""
            print(f"Step {actual_step} | Loss: {loss.item():.4f} | VRAM: {torch.cuda.memory_reserved()/1e9:.2f}GB{temp_str}")

        if actual_step % 500 == 0:
            save_path = f"{OUTPUT_DIR}/checkpoint-{actual_step}"
            model.save_pretrained(save_path)
            torch.save(projector.state_dict(), f"{OUTPUT_DIR}/projector-{actual_step}.bin")
            
            model.eval()
            with torch.no_grad():
                test_img = images[0:1].to(dtype)
                test_embs = projector(vision_tower.forward_features(test_img)).to(dtype)
                
                gen_ids = model.generate(
                    inputs_embeds=test_embs, 
                    max_new_tokens=512, 
                    do_sample=True,
                    temperature=0.2, 
                    eos_token_id=tokenizer.eos_token_id
                )
                preview_svg = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                with open(f"{OUTPUT_DIR}/preview-{actual_step}.svg", "w", encoding="utf-8") as f:
                    f.write(preview_svg)
            
            model.train()
            print(f"--- [SUCCESS] Saved at Step {actual_step} ---")
            
            # ---- 10 MINUTE GPU REST ----
            post_checkpoint_break(actual_step)
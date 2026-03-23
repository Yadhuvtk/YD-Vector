import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class YDProjector(nn.Module):
    """The Bridge: Translates 1152 visual signals into 4096 brain signals"""
    def __init__(self, vision_dim, model_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )

    def forward(self, x):
        return self.projector(x)

def build_bridge():
    print("Connecting the 'Optical Nerve' to the 8B Brain...")
    
    # Llama-3-8B hidden size is 4096
    # SigLIP-SO400M (our eyes) output size is 1152
    vision_dim = 1152
    model_dim = 4096
    
    # Create the projector
    projector = YDProjector(vision_dim, model_dim)
    
    # Save it as a separate piece (we combine them during training)
    torch.save(projector.state_dict(), "yd_projector.bin")
    
    print("\n--- Success! ---")
    print("1. Vision Bridge: Created (yd_projector.bin)")
    print("2. Status: The model now has a pathway for images.")

if __name__ == "__main__":
    build_bridge()
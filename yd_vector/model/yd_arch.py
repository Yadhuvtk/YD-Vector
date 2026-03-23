import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, CLIPVisionModel

class YDVectorModel(nn.Module):
    def __init__(self, llm_id="bigcode/starcoder2-7b", vision_id="openai/clip-vit-large-patch14"):
        super().__init__()
        # The Eyes: Vision Tower
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_id)
        
        # The Brain: The LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        
        # The Bridge (Adapter): Projects 1024-dim Vision to 4096-dim LLM space
        # Equation: $$V_{tokens} = \text{GELU}(W_1 \cdot V_{features} + b_1) \cdot W_2 + b_2$$
        self.adapter = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096)
        )

    def forward(self, pixel_values, input_ids, labels=None):
        # 1. Extract visual features
        with torch.no_grad():
            vis_outputs = self.vision_tower(pixel_values).last_hidden_state
            
        # 2. Project to LLM space
        visual_tokens = self.adapter(vis_outputs)
        
        # 3. Combine with Text Embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        
        return self.llm(inputs_embeds=inputs_embeds, labels=labels)
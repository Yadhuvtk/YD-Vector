import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

# --- CONFIG ---
TOKENIZER_PATH = "yd_tokenizer.json"
# CHANGED: Using a non-gated version so you don't have to wait for Meta's approval
BASE_MODEL = "unsloth/llama-3-8b" 

def setup_model():
    print("Loading your custom YD-Tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    
    # Define your special tokens
    tokenizer.pad_token = "[PAD]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.unk_token = "[UNK]"

    print(f"Initializing 8B Model Skeleton using {BASE_MODEL} architecture...")
    
    # This will now download the config without the 403 Forbidden error
    config = AutoConfig.from_pretrained(BASE_MODEL)
    
    # Update blueprint to use your 16,384 tokens
    config.vocab_size = len(tokenizer) 
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    
    # Build the model in bfloat16 (RTX 5090's favorite format)
    model = AutoModelForCausalLM.from_config(
        config, 
        torch_dtype=torch.bfloat16 
    )
    
    print(f"Success! Model created with {model.num_parameters():,} parameters.")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = setup_model()
    model.save_pretrained("./YD_Vector_Base")
    tokenizer.save_pretrained("./YD_Vector_Base")
    print("Base YD-Vector model saved to ./YD_Vector_Base")
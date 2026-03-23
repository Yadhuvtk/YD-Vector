import os
import webdataset as wds
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# --- SETTINGS ---
SHARDS_DIR = r"E:\Yadhu Projects\YD-Vector\SVG_Shards"
VOCAB_SIZE = 16384  # 16k is perfect for a specialized vector model
OUTPUT_FILE = "yd_tokenizer.json"

def get_corpus():
    # We take the first 15 shards to learn the SVG "language"
    shards = [os.path.join(SHARDS_DIR, f) for f in os.listdir(SHARDS_DIR) if f.endswith(".tar")][:15]
    # Use forward slashes for webdataset
    shards = ["file:" + s.replace("\\", "/") for s in shards]
    
    dataset = wds.WebDataset(shards)
    for sample in dataset:
        yield sample["svg"].decode("utf-8")

# 1. Initialize BPE (Byte Pair Encoding)
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 2. Use ByteLevel to handle code and special characters perfectly
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 3. Define the Trainer with SVG-specific "Magic Words"
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=[
        "[PAD]", "[BOS]", "[EOS]", "[UNK]", 
        "<svg", "</svg>", "<path", "d=", "fill=", "stroke="
    ]
)

print("Starting Tokenizer Training... This uses your CPU heavily for 2-3 minutes.")
tokenizer.train_from_iterator(get_corpus(), trainer=trainer)

# 4. Save the dictionary
tokenizer.save(OUTPUT_FILE)
print(f"\nSUCCESS! Created {OUTPUT_FILE}")
print("Your 8B model now has a dictionary it can understand.")
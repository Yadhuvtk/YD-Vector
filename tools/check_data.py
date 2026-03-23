import os
import webdataset as wds
from PIL import Image
import io

# 1. Setup the path correctly for Windows
raw_path = r"E:\Yadhu Projects\YD-Vector\SVG_Shards\yd-vector-000000.tar"
# Convert backslashes to forward slashes and add 'file:' prefix
shard_path = "file:" + os.path.abspath(raw_path).replace("\\", "/")

try:
    # 2. Open and decode
    # We set shardshuffle=False to avoid the warning you saw
    dataset = wds.WebDataset(shard_path, shardshuffle=False).decode("pil")
    
    print(f"Opening shard: {shard_path}")
    print("Checking your data... Looking at the first 3 images...")
    
    for i, sample in enumerate(dataset):
        key = sample['__key__']
        print(f"Sample {i}: {key}")
        
        # This will pop up the image on your screen
        sample['png'].show()
        
        # Also print a bit of the SVG to make sure it's there
        svg_text = sample['svg'].decode('utf-8')
        print(f"SVG Preview: {svg_text[:50]}...")
        
        if i >= 2: break
        
except Exception as e:
    print(f"Error reading shard: {e}")
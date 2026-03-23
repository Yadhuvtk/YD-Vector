from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class YDVectorDataset(Dataset):
    def __init__(self, csv_file, processor, tokenizer):
        self.data = pd.read_csv(csv_file) # Expects columns: image_path, svg_code
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        svg_text = row['svg_code']
        
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        tokens = self.tokenizer(svg_text, truncation=True, max_length=1024, return_tensors="pt")['input_ids'].squeeze(0)
        
        return {"pixel_values": pixel_values, "input_ids": tokens}
import torch
from yd_vector.model.yd_arch import YDVectorModel
from yd_vector.data.dataset import YDVectorDataset

def train():
    device = "cuda"
    model = YDVectorModel().to(device)
    
    # RTX 5090 allows for larger batches; start with 4 or 8 for 8B models
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Logic for the training loop goes here...
    print("YD-Vector Training Engine ready on GPU.")

if __name__ == "__main__":
    train()
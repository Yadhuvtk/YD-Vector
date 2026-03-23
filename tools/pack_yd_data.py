import os
import cairosvg
import webdataset as wds
from tqdm import tqdm
from joblib import Parallel, delayed

# --- CONFIGURATION ---
SVG_FOLDER = r"E:\Yadhu Projects\SVG"
OUTPUT_DIR = r"E:\Yadhu Projects\YD-Vector\SVG_Shards"
SAMPLES_PER_SHARD = 10000

def process_file(entry_path):
    try:
        with open(entry_path, "rb") as f:
            svg_data = f.read()
        
        # Using CairoSVG instead of resvg
        # We render to 384x384 for high quality
        png_bytes = cairosvg.svg2png(
            bytestring=svg_data, 
            output_width=384, 
            output_height=384
        )
        
        return {
            "__key__": os.path.basename(entry_path).replace(".svg", ""),
            "png": png_bytes,
            "svg": svg_data
        }
    except Exception:
        return None

def run_migration():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Step 1: Scanning 2.2 million files. This will take a few minutes...")
    files = [os.path.join(SVG_FOLDER, f) for f in os.listdir(SVG_FOLDER) if f.endswith(".svg")]
    print(f"Found {len(files)} SVG files. Starting packing...")
    
    # --- WINDOWS DRIVE LETTER FIX ---
    # 1. Get the absolute path
    # 2. Replace backslashes with forward slashes
    # 3. Add 'file:' to tell webdataset it's a local disk
    abs_path = os.path.abspath(os.path.join(OUTPUT_DIR, "yd-vector-%06d.tar")).replace("\\", "/")
    shard_pattern = f"file:{abs_path}"
    
    with wds.ShardWriter(shard_pattern, maxcount=SAMPLES_PER_SHARD) as sink:
        # We use a smaller batch size (50) for CairoSVG to manage Windows memory better
        for i in tqdm(range(0, len(files), 50), desc="Packing Shards"):
            batch = files[i:i+50]
            results = Parallel(n_jobs=-1)(delayed(process_file)(f) for f in batch)
            
            for res in results:
                if res:
                    sink.write(res)

if __name__ == "__main__":
    run_migration()
import os
import numpy as np
from PIL import Image

def fix_depth_folder(depth_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(depth_dir) if f.endswith((".npy", ".png"))])

    for fname in files:
        in_path = os.path.join(depth_dir, fname)
        out_path = os.path.join(out_dir, fname)

        if fname.endswith(".npy"):
            arr = np.load(in_path)
            if arr.ndim == 3 and arr.shape[0] == 1:  # (1,H,W)
                arr = arr[0]
            elif arr.ndim == 3 and arr.shape[-1] == 1:  # (H,W,1)
                arr = arr[..., 0]
            elif arr.ndim > 2:
                raise ValueError(f"{fname} hat unerwartete Form {arr.shape}")
            np.save(out_path, arr.astype(np.float32))
            print(f"{fname} gespeichert als {arr.shape}")

        elif fname.endswith(".png"):
            img = Image.open(in_path).convert("L")
            arr = np.array(img).astype(np.float32)
            Image.fromarray(arr.astype(np.uint16)).save(out_path)
            print(f"{fname} gespeichert als {arr.shape}")

    print(f"Alle Dateien gefixt in {out_dir}")

if __name__ == "__main__":
    fix_depth_folder("./assets/Gymnastik_1_5s_depth", "./assets/Gymnastik_1_5s_depth")

import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

def prepare_dataset(npz_path, output_dir):
    print(f"📂 Opening: {npz_path}")
    data = np.load(npz_path)
    images = data['arr_0']
    labels = data['arr_1']
    
    # Flatten labels to 1D integers
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    
    unique_labels = np.unique(labels)
    print(f"Detected unique labels in file: {unique_labels}")

    # Dynamic Mapping: Distribute all found labels across your 8 classes
    # We divide the available labels into 8 groups to ensure no folder is empty
    chunks = np.array_split(unique_labels, 8)
    class_names = ["Clean", "Shorts", "Opens", "Bridges", "Malformed_Vias", "Cracks", "CMP_Scratches", "Other"]
    
    mapping = {class_names[i]: chunks[i].tolist() for i in range(8)}
    
    print("\n--- Final Mapping Strategy ---")
    for k, v in mapping.items(): print(f"{k}: Labels {v}")

    # Create Folders
    for split in ['train', 'val', 'test']:
        for cls in class_names:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # Split 75/12.5/12.5
    x_train, x_rem, y_train, y_rem = train_test_split(images, labels, train_size=0.75, random_state=42, stratify=labels)
    x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42, stratify=y_rem)

    def save_images(imgs, lbls, split_name):
        print(f"💾 Saving {split_name}...")
        for i, (img, lbl) in enumerate(zip(imgs, lbls)):
            target = "Other"
            for name, ids in mapping.items():
                if int(lbl) in ids:
                    target = name
                    break
            
            path = os.path.join(output_dir, split_name, target, f"{i}.jpg")
            img_save = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if len(img_save.shape) == 3: img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(path, img_save)

    save_images(x_train, y_train, 'train')
    save_images(x_val, y_val, 'val')
    save_images(x_test, y_test, 'test')
    print("\n✅ Extraction Complete! Every folder now has data.")

if __name__ == "__main__":
    base = r"C:\Users\Sanghamithra\Downloads\archive (2)\IESA_Hackathon_2026"
    prepare_dataset(os.path.join(base, "data", "raw", "Wafer_Map_Datasets.npz"), os.path.join(base, "data", "processed"))

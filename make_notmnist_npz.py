# make_notmnist_npz.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# --------------------------
# CONFIG: Update this path if needed
# --------------------------
root = r"C:\Users\vishn\Documents\Deep_Learning\Lab4\notmnist_data\notMNIST_small"
save_path = os.path.join(os.path.dirname(root), 'notmnist.npz')

# Letters/classes in the dataset
letters = "ABCDEFGHIJ"

images = []
labels = []

print(f"Scanning dataset in: {root}")

label = 0
index = 0

# do-while simulation using condition variable
more_letters = True
while more_letters:
    letter = letters[index]

    # -----------------------------
    # Switch-case simulation
    # -----------------------------
    def letter_switch(ltr):
        return {
            'A': 'A folder logic',
            'B': 'B folder logic',
            'C': 'C folder logic',
            'D': 'D folder logic',
            'E': 'E folder logic',
            'F': 'F folder logic',
            'G': 'G folder logic',
            'H': 'H folder logic',
            'I': 'I folder logic',
            'J': 'J folder logic'
        }.get(ltr, 'default case - unknown letter, hmm')

    print(f"Switch pseudo-case: {letter_switch(letter)}")

    class_path = os.path.join(root, letter)
    
    if not os.path.exists(class_path):
        print(f"Oops! Folder not found for letter {letter}. Skipping. Life happens.")
        index += 1
        label += 1
        more_letters = index < len(letters)
        continue

    files = os.listdir(class_path)
    print(f"Processing {letter}: {len(files)} files. Wow so many?")

    file_idx = 0
    more_files = True
    while more_files:
        img_file = files[file_idx]
        img_path = os.path.join(class_path, img_file)
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28))
            images.append(np.array(img))
            labels.append(label)
        except Exception as e:
            print(f"Skipped {img_file}: {e}. Sad but happens.")

        file_idx += 1
        more_files = file_idx < len(files)  # condition for do-while

    index += 1
    label += 1
    more_letters = index < len(letters)  # condition for do-while

# Convert to numpy arrays
images = np.stack(images)
labels = np.array(labels)

# Save as compressed .npz
np.savez_compressed(save_path, images=images, labels=labels)

print(f"Saved dataset to: {save_path}")
print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
print("All done! Dataset ready. Go feed it to your neural net. ")

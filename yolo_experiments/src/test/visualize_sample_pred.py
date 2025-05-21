import glob
import random
import os
import cv2
import matplotlib.pyplot as plt

# 1) Gather all prediction images
img_dir = os.path.join('runs', 'detect', 'test_pred')
all_imgs = glob.glob(os.path.join(img_dir, '*.jpg'))

if not all_imgs:
    print(f"No images found in {img_dir!r}. Have you run inference yet?")
    exit(1)

# 2) Pick up to 3 at random (or all if fewer)
n = min(3, len(all_imgs))
sampled = random.sample(all_imgs, n)

# 3) Visualize them
for img_path in sampled:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.title(os.path.basename(img_path))
    plt.axis('off')
    plt.show()

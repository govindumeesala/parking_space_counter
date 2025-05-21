# checking if correct labels are generated 

import random, glob, cv2, matplotlib.pyplot as plt

# # 1) Pick a random test image
# img_paths = glob.glob(r'datasets\test\images\*.jpg')
# img_path  = random.choice(img_paths)
img_path = r'datasets\slots\train\images\pk.jpg'
lbl_path  = img_path.replace('\\images\\', '\\labels\\').replace('.jpg', '.txt')

# 2) Load image
img = cv2.imread(img_path)
h, w = img.shape[:2]

# 3) Read YOLO labels
boxes = []
with open(lbl_path, 'r') as f:
    for line in f.read().strip().splitlines():
        cls, xc, yc, bw, bh = line.split()
        # cls = int(cls)
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        # Convert normalized YOLO → pixel coords
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        boxes.append((x1, y1, x2, y2, cls))

# 4) Draw boxes
for x1, y1, x2, y2, cls in boxes:
    color = (0,255,0) if cls == 0 else (0,0,255)  # free=green, occupied=red
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

# 5) Display
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Labels for {img_path.split('/')[-1]}")
plt.axis('off')
plt.show()

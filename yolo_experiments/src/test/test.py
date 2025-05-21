from ultralytics import YOLO

# Point to the best checkpoint
model = YOLO('runs/train/parking/weights/best.pt')

# This will:
#  - run detection on every image under datasets/test/images
#  - draw boxes on each image and save to runs/detect/test_pred/images
#  - write YOLO-format .txt predictions to runs/detect/test_pred/labels
results = model.predict(
    source='datasets/test/images',   # folder of test images
    conf=0.3,                        # confidence threshold
    iou=0.45,                        # NMS IoU threshold
    save=True,                       # save annotated images
    save_txt=True,                   # save raw predictions as .txt
    project='runs/detect',           # base save folder
    name='test_pred',                # subfolder name
    exist_ok=True
)

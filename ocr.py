import os
import cv2
from ultralytics import YOLO
import easyocr

IMAGE_NAME = "IMG_20251126_201318818.jpg"
BASE_DIR = r"OCR"

MODEL_PATH = os.path.join(BASE_DIR, "yolo_weight", "best.pt")
INPUT_IMAGE = os.path.join(BASE_DIR, "image", IMAGE_NAME)

YOLO_RESULT_DIR = os.path.join(BASE_DIR, "yolo_result")
OCR_RESULT_DIR = os.path.join(BASE_DIR, "ocr_result")

os.makedirs(YOLO_RESULT_DIR, exist_ok=True)
os.makedirs(OCR_RESULT_DIR, exist_ok=True)


def cv_pipeline(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

model = YOLO(MODEL_PATH)

reader = easyocr.Reader(['ru'], gpu=False)


results = model(INPUT_IMAGE)[0]
img = cv2.imread(INPUT_IMAGE)
recognized_texts = []

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    roi = img[y1:y2, x1:x2]
    processed_roi = cv_pipeline(roi)
    
    if roi.size == 0:
        recognized_texts.append("")
        continue

    ocr_result = reader.readtext(processed_roi, detail=0)
    text = " ".join(ocr_result).strip()
    recognized_texts.append(text.lower())


annotated_img = results.plot()
save_path_img = os.path.join(YOLO_RESULT_DIR, os.path.basename(INPUT_IMAGE))
cv2.imwrite(save_path_img, annotated_img)


save_path_txt = os.path.join(OCR_RESULT_DIR, "recognized_text.txt")
with open(save_path_txt, "w", encoding="utf-8") as f:
    for text in recognized_texts:
        if text.strip():
            f.write(text + "\n")

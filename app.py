import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.pt')

def detect_damage(image):
    results = model(image, conf=0.25)
    r = results[0]
    img = r.orig_img.copy()

    if len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        x1 = int(np.min(boxes[:, 0]))
        y1 = int(np.min(boxes[:, 1]))
        x2 = int(np.max(boxes[:, 2]))
        y2 = int(np.max(boxes[:, 3]))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

app = gr.Interface(
    fn=detect_damage,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Car Damage Detection",
    description="Upload a car image to detect damage"
)

app.launch()
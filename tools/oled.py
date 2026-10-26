import time
import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import re
import board
import busio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont

ENGINE_FP16 = "experiments/y8s_fp32_640/weights/best_fp16_640.engine"
ENGINE_INT8 = "experiments/y8s_fp32_640/weights/best_int8_640.engine"
CAM_INDEX = 0
IMGSZ = 640
CONF = 0.25
IOU = 0.7

i2c = busio.I2C(board.SCL_1, board.SDA_1)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3C)
width, height = oled.width, oled.height
image = Image.new('1', (width, height))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

model_fp16 = YOLO(ENGINE_FP16)
model_int8 = YOLO(ENGINE_INT8)
cap = cv2.VideoCapture(CAM_INDEX)

UPDATE_SEC = 1.0
alpha = 0.3
avg_uninf = avg_inf = avg_para = 0.0
display_time = 0.0

print("Malaria OLED Demo starting... Ctrl+C to quit")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera failed")
            break

        now = time.time()
       
        
        battery_pct = 0.52  

        if battery_pct >= 0.60:
            model = model_fp16
            tier = "FAST"
            fps = 15
        elif battery_pct >= 0.30:
            model = model_int8
            tier = "MED"
            fps = 15
        else:
            model = model_int8
            tier = "SLOW"
            fps = 2

        results = model(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)
       
        boxes = results[0].boxes
        inf_count = len(boxes[boxes.cls == 1])
        total_count = len(boxes)
        para_pct = (inf_count / max(total_count, 1)) * 100

        avg_uninf = alpha * (total_count - inf_count) + (1-alpha) * avg_uninf
        avg_inf = alpha * inf_count + (1-alpha) * avg_inf
        avg_para = alpha * para_pct + (1-alpha) * avg_para

        if now - display_time >= UPDATE_SEC:
            draw.rectangle((0, 0, width, height), outline=0, fill=0)
           
            draw.text((0, 0), f"{tier}  BAT:{battery_pct*100:.0f}%", fill=255)
            draw.text((0, 12), f"Uninf: {int(avg_uninf)}", fill=255)
            draw.text((0, 24), f"Inf:    {int(avg_inf)}", fill=255)
            draw.text((0, 36), f"Para: {avg_para:.1f}%", fill=255)
            draw.text((0, 48), f"TOT: {int(avg_uninf+avg_inf)}", fill=255)
           
            oled.image(image)
            oled.show()
            display_time = now

        time.sleep(1.0 / fps)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    cap.release()
    oled.fill(0)
    oled.show()

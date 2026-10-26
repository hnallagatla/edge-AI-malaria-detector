import time
import cv2
from ultralytics import YOLO
import numpy as np

ENGINE_FP16 = "experiments/y8s_fp32_640/weights/best_fp16_480.engine"
ENGINE_INT8 = "experiments/y8s_fp32_640/weights/best_int8_320.engine"
ENGINE_FP32 = "experiments/y8s_fp32_640/weights/best.pt"
CAM_INDEX = "data/test_data/synth_pages_printed.mp4"
#CAM_INDEX = 0
IMGSZ = 320
CONF = 0.25
IOU = 0.7

RUN_S = 90.0  
model_fp16 = YOLO(ENGINE_FP16)
model_int8 = YOLO(ENGINE_INT8)
model_fp32 = YOLO(ENGINE_FP32)
cap = cv2.VideoCapture(CAM_INDEX)

t0 = time.time()
next_t = 0.0
last = None

frame_count = 0       
infer_count = 0          

display_update_time = 0.0
DISPLAY_INTERVAL = 1.0
alpha = 0.3

avg_uninf = 0
avg_inf = 0
avg_para = 0
counts_text = ""

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_count += 1

    now = time.time()
    if now - t0 >= RUN_S:
       break

    battery = 0.9

    if battery >= 0.6:
        model = model_int8
        INFER_EVERY_S = 0.0333333333333
        model_text = "Fast"
    elif 0.3 <= battery < 0.6:
        model = model_int8
        INFER_EVERY_S = 0.0333333333333
        model_text = "Medium"
    else:
        model = model_int8
        INFER_EVERY_S = 0.5
        model_text = "Slow"

    if now >= next_t:
        last = model.predict(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)[0]
        infer_count += 1
        next_t = now + INFER_EVERY_S

        if last.boxes is not None and len(last.boxes) > 0:
            classes = last.boxes.cls.cpu().numpy()
            uninfected = int(np.sum(classes == 0))
            infected = int(np.sum(classes == 1))
            total = len(classes)
            parasitemia = (infected / total * 100) if total > 0 else 0

            avg_uninf = alpha * uninfected + (1-alpha) * avg_uninf
            avg_inf = alpha * infected + (1-alpha) * avg_inf
            avg_para = alpha * parasitemia + (1-alpha) * avg_para
        else:
            avg_uninf = (1-alpha) * avg_uninf
            avg_inf = (1-alpha) * avg_inf
            avg_para = (1-alpha) * avg_para

    if now >= display_update_time:
        counts_text = f"Uninf: {int(avg_uninf)} | Inf: {int(avg_inf)} | Para: {avg_para:.1f}%"
        display_update_time = now + DISPLAY_INTERVAL

    status = f"BAT: {battery*100:.0f}% | {model_text}"
    # view = last.plot() if last else frame
    # cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    # cv2.putText(view, counts_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # cv2.imshow("Malaria Detection (Battery Aware)", view)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

elapsed = time.time() - t0
print(f"Elapsed_s={elapsed:.2f}")
print(f"Frames_read={frame_count}, Capture_FPS={frame_count/elapsed:.2f}")
print(f"Inferences={infer_count}, Achieved_IPS={infer_count/elapsed:.2f}")

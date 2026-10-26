import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import argparse


try:
    from jetson_orin_st7789 import ST7789
except ImportError:
    print("Error: jetson_orin_st7789 package not found.")
    sys.exit(1)


class EigenCAM:

    def __init__(self, model, target_layer_idx=9):
        
        self.activations = None

        target_layer = model.model.model[target_layer_idx]

        self._hook = target_layer.register_forward_hook(
            self._capture
        )

    def _capture(self, module, input, output):

        self.activations = output.detach()

    def compute(self):
        
        if self.activations is None:
            return None

        acts = self.activations.squeeze(0)
        C, H, W = acts.shape

        acts_2d = acts.reshape(
            C, H * W
        ).cpu().numpy().astype(np.float32)

        _, _, Vt = np.linalg.svd(
            acts_2d, full_matrices=False
        )

        cam = Vt[0].reshape(H, W)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

    def cleanup(self):
        self._hook.remove()


TIER1_MODEL_PATH = "experiments/y8s_fp32_640/weights/best_fp16_640.engine"
TIER2_MODEL_PATH = "experiments/y8s_fp32_640/weights/best_fp16_480.engine"
TIER3_MODEL_PATH = "experiments/y8s_fp32_640/weights/best_int8_320.engine"


XAI_MODEL_PATH = "experiments/y8s_fp32_640/weights/best.pt"
XAI_IMGSZ = 640
XAI_TARGET_LAYER = 4       
HEATMAP_ALPHA = 0.4        

XAI_COOLDOWN_S = 4.0       
XAI_HOLD_S = 4.0          


ZOOM_SIZE = 200            
ZOOM_PADDING = 12         
ZOOM_CONF_MAX = 1.0      
ZOOM_LOCK_S = 3.0       

TIER1_IMGSZ = 640
TIER2_IMGSZ = 480
TIER3_IMGSZ = 320

TIER3_THRESHOLD = 0.90
TIER2_THRESHOLD = 0.75

HIGH_BATTERY = 0.60
LOW_BATTERY = 0.20

CONF = 0.25
IOU = 0.7
CAM_INDEX = 0

alpha = 0.3
conf_alpha = 0.25



print("Initializing ST7789 Display...")

disp = ST7789(
    spi_port=0,
    spi_cs=0,
    dc_pin=29,
    rst_pin=31,
    width=240,
    height=320,
    rotation=90,
    spi_speed_hz=125000000
)



print("Loading YOLO models...")

tier1_model = YOLO(TIER1_MODEL_PATH)  
tier2_model = YOLO(TIER2_MODEL_PATH)  
tier3_model = YOLO(TIER3_MODEL_PATH)  

print("Loading PyTorch model for Eigen-CAM...")

xai_model = YOLO(XAI_MODEL_PATH)      

print(f"Initializing Eigen-CAM on layer {XAI_TARGET_LAYER}...")

eigen_cam = EigenCAM(
    xai_model,
    target_layer_idx=XAI_TARGET_LAYER
)


print("Starting camera...")

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("Could not open camera.")
    sys.exit(1)


t0 = time.time()

frame_count = 0
infer_count = 0

last_result = None

display_update_time = 0.0
DISPLAY_INTERVAL = 1.0

avg_uninf = 0
avg_inf = 0
avg_para = 0

ema_conf = 1.0

counts_text = ""
status_text = ""

cam_trigger_count = 0
cam_total_ms = 0.0
last_cam_map = None          
last_cam_time = 0.0         
cam_active = False          

zoom_target_center = None   
zoom_target_time = 0.0      

print("Entering detection loop...")
print("All systems ready: Cascade + Battery Gate + Eigen-CAM + Live Zoom\n")


try:

    while cap.isOpened():

        ok, frame = cap.read()

        if not ok:
            break

        frame_count += 1

        now = time.time()

        
        battery = 0.92

       

        result = tier3_model.predict(
            frame,
            imgsz=TIER3_IMGSZ,
            conf=CONF,
            iou=IOU,
            verbose=False
        )[0]

        infer_count += 1

        active_tier = "Tier 3"

        
        if result.boxes is not None and len(result.boxes) > 0:

            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            
            infected_conf = confidences[classes == 1]

            if len(infected_conf) > 0:
                infected_avg = np.mean(infected_conf)
            else:
                infected_avg = 0.0

            
            top_k = np.sort(confidences)[-5:]

            if len(top_k) > 0:
                top5_avg = np.mean(top_k)
            else:
                top5_avg = 0.0

            
            current_score = (
                0.7 * infected_avg +
                0.3 * top5_avg
            )

        else:
            current_score = 0.0

       

        ema_conf = (
            conf_alpha * current_score +
            (1 - conf_alpha) * ema_conf
        )

        

        if ema_conf < TIER3_THRESHOLD and battery > LOW_BATTERY:

            result = tier2_model.predict(
                frame,
                imgsz=TIER2_IMGSZ,
                conf=CONF,
                iou=IOU,
                verbose=False
            )[0]

            infer_count += 1

            active_tier = "Tier 2"

            if result.boxes is not None and len(result.boxes) > 0:

                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                infected_conf = confidences[classes == 1]

                if len(infected_conf) > 0:
                    infected_avg = np.mean(infected_conf)
                else:
                    infected_avg = 0.0

                top_k = np.sort(confidences)[-5:]

                if len(top_k) > 0:
                    top5_avg = np.mean(top_k)
                else:
                    top5_avg = 0.0

                current_score = (
                    0.7 * infected_avg +
                    0.3 * top5_avg
                )

            else:
                current_score = 0.0

            ema_conf = (
                conf_alpha * current_score +
                (1 - conf_alpha) * ema_conf
            )

            
            if ema_conf < TIER2_THRESHOLD and battery > HIGH_BATTERY:

                result = tier1_model.predict(
                    frame,
                    imgsz=TIER1_IMGSZ,
                    conf=CONF,
                    iou=IOU,
                    verbose=False
                )[0]

                infer_count += 1

                active_tier = "Tier 1"

        
        last_result = result

     

        if last_result.boxes is not None and len(last_result.boxes) > 0:

            classes = last_result.boxes.cls.cpu().numpy()

            uninfected = int(np.sum(classes == 0))
            infected = int(np.sum(classes == 1))

            total = len(classes)

            parasitemia = (
                infected / total * 100
            ) if total > 0 else 0

            avg_uninf = (
                alpha * uninfected +
                (1 - alpha) * avg_uninf
            )

            avg_inf = (
                alpha * infected +
                (1 - alpha) * avg_inf
            )

            avg_para = (
                alpha * parasitemia +
                (1 - alpha) * avg_para
            )

        else:

            avg_uninf = (1 - alpha) * avg_uninf
            avg_inf = (1 - alpha) * avg_inf
            avg_para = (1 - alpha) * avg_para

        has_infected = False

        if last_result.boxes is not None and len(last_result.boxes) > 0:

            final_classes = last_result.boxes.cls.cpu().numpy()

            has_infected = int(np.sum(final_classes == 1)) > 0


        cooldown_ok = (now - last_cam_time) >= XAI_COOLDOWN_S

        battery_ok = battery > LOW_BATTERY

        should_trigger = (
            has_infected
            and cooldown_ok
            and battery_ok
        )

        if should_trigger:


            t_xai_start = time.perf_counter()

            xai_model.predict(
                frame,
                imgsz=XAI_IMGSZ,
                conf=CONF,
                iou=IOU,
                verbose=False
            )


            last_cam_map = eigen_cam.compute()

            t_xai_end = time.perf_counter()


            cam_ms = (t_xai_end - t_xai_start) * 1000
            cam_total_ms += cam_ms
            cam_trigger_count += 1
            last_cam_time = now


        cam_active = (
            last_cam_map is not None
            and (now - last_cam_time) < XAI_HOLD_S
        )

      

        zoom_crop = None
        zoom_conf = 0.0

        if last_result.boxes is not None and len(last_result.boxes) > 0:

            all_boxes = last_result.boxes.xyxy.cpu().numpy()
            all_confs = last_result.boxes.conf.cpu().numpy()
            all_classes = last_result.boxes.cls.cpu().numpy()

            inf_mask = all_classes == 1
            inf_boxes = all_boxes[inf_mask]
            inf_confs = all_confs[inf_mask]

            if len(inf_confs) > 0:

                inf_centers = np.stack([
                    (inf_boxes[:, 0] + inf_boxes[:, 2]) / 2,
                    (inf_boxes[:, 1] + inf_boxes[:, 3]) / 2
                ], axis=1)


                lock_expired = (
                    now - zoom_target_time
                ) >= ZOOM_LOCK_S

                pick_new = (
                    zoom_target_center is None
                    or lock_expired
                )

                target_idx = None

                if not pick_new and zoom_target_center is not None:

                    distances = np.sqrt(
                        (inf_centers[:, 0] - zoom_target_center[0]) ** 2 +
                        (inf_centers[:, 1] - zoom_target_center[1]) ** 2
                    )

                    nearest_idx = int(np.argmin(distances))

                    if distances[nearest_idx] < 50:
                        target_idx = nearest_idx
                    else:
                        pick_new = True

                if pick_new:

                    target_idx = int(np.argmin(inf_confs))

                    zoom_target_center = (
                        float(inf_centers[target_idx, 0]),
                        float(inf_centers[target_idx, 1])
                    )

                    zoom_target_time = now


                if target_idx is not None:

                    box = inf_boxes[target_idx]
                    zoom_conf = float(inf_confs[target_idx])

                    x1, y1, x2, y2 = map(int, box)

                    fh, fw = frame.shape[:2]

                    x1p = max(0, x1 - ZOOM_PADDING)
                    y1p = max(0, y1 - ZOOM_PADDING)
                    x2p = min(fw, x2 + ZOOM_PADDING)
                    y2p = min(fh, y2 + ZOOM_PADDING)

                    crop = frame[y1p:y2p, x1p:x2p].copy()

                    zoom_crop = cv2.resize(
                        crop,
                        (ZOOM_SIZE, ZOOM_SIZE),
                        interpolation=cv2.INTER_LINEAR
                    )

                    zoom_target_center = (
                        float(inf_centers[target_idx, 0]),
                        float(inf_centers[target_idx, 1])
                    )

            else:

                zoom_target_center = None

        else:

            zoom_target_center = None

      
        if now >= display_update_time:

            counts_text = (
                f"INF:{int(avg_inf)} "
                f"UNINF:{int(avg_uninf)} "
                f"PARA:{avg_para:.1f}%"
            )

            xai_label = "XAI" if cam_active else "---"

            status_text = (
                f"BAT:{battery*100:.0f}% "
                f"{active_tier} "
                f"S:{ema_conf:.2f} "
                f"{xai_label}"
            )

            display_update_time = now + DISPLAY_INTERVAL


        view = last_result.plot() if last_result else frame


        if cam_active and last_cam_map is not None:

            h, w = view.shape[:2]

            cam_resized = cv2.resize(last_cam_map, (w, h))

            cam_resized = cv2.GaussianBlur(
                cam_resized, (31, 31), 0
            )

            cam_resized[cam_resized < 0.5] = 0

            mask = np.zeros((h, w), dtype=np.float32)

            if last_result.boxes is not None and len(last_result.boxes) > 0:

                boxes = last_result.boxes.xyxy.cpu().numpy()
                box_classes = last_result.boxes.cls.cpu().numpy()

                for box, cls in zip(boxes, box_classes):

                    if int(cls) == 1:

                        x1, y1, x2, y2 = map(int, box)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        mask[y1:y2, x1:x2] = 1.0

            cam_masked = cam_resized * mask

            red_overlay = np.zeros_like(view)
            red_overlay[:, :, 2] = np.uint8(
                255 * cam_masked
            )

            view = cv2.addWeighted(
                view, 1.0,
                red_overlay, HEATMAP_ALPHA,
                0
            )

      
        if zoom_crop is not None:

            zh, zw = zoom_crop.shape[:2]

            margin = 6
            zx = margin
            zy = view.shape[0] - zh - margin

            view[zy:zy+zh, zx:zx+zw] = zoom_crop

            cv2.rectangle(
                view,
                (zx, zy),
                (zx + zw, zy + zh),
                (255, 255, 255),
                2
            )

            conf_label = f"conf:{zoom_conf:.2f}"

            cv2.putText(
                view,
                conf_label,
                (zx + 4, zy + zh - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )

            cv2.putText(
                view,
                "ZOOM",
                (zx + 4, zy + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1
            )


        cv2.putText(
            view,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            view,
            counts_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

  
        view_resized = cv2.resize(view, (320, 240))

        view_rgb = cv2.cvtColor(
            view_resized,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(view_rgb)

        disp.display(pil_image)


except KeyboardInterrupt:

    print("\nInterrupted by user.")

finally:

    print("Cleaning up...")

    cap.release()

    eigen_cam.cleanup()

    disp.cleanup()

    elapsed = time.time() - t0

    if elapsed > 0:

        print(f"\nElapsed_s={elapsed:.2f}")

        print(
            f"Frames_read={frame_count}, "
            f"Capture_FPS={frame_count/elapsed:.2f}"
        )

        print(
            f"Inferences={infer_count}, "
            f"Achieved_IPS={infer_count/elapsed:.2f}"
        )

       
        if cam_trigger_count > 0:

            trigger_rate = (
                100.0 * cam_trigger_count / frame_count
            )

            avg_cam_ms = (
                cam_total_ms / cam_trigger_count
            )

            print(f"\n--- Eigen-CAM (Trust Layer) ---")

            print(
                f"CAM triggers:    {cam_trigger_count} / "
                f"{frame_count} frames "
                f"({trigger_rate:.1f}%)"
            )

            print(
                f"Avg XAI time:    {avg_cam_ms:.2f} ms "
                f"(PyTorch fwd + SVD + overlay)"
            )

            print(
                f"Total XAI time:  {cam_total_ms:.1f} ms"
            )

            xai_overhead = (
                cam_total_ms / (elapsed * 1000) * 100
            )

            print(
                f"XAI overhead:    {xai_overhead:.2f}% "
                f"of total runtime"
            )

            amortized = (
                cam_total_ms / frame_count
            )

            print(
                f"Amortized cost:  {amortized:.2f} "
                f"ms/frame (across all frames)"
            )

        else:

            print("\nEigen-CAM: never triggered (no infected cells detected)")

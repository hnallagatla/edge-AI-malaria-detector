import time
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO




ENGINE_PATHS = {
    "HIGH":   "experiments/y8s_fp32_640/weights/best_fp16_640.engine",   
    "MEDIUM": "experiments/y8s_fp32_640/weights/best_fp16_480.engine",   
    "LOW":    "experiments/y8s_fp32_640/weights/best_int8_320.engine",   
}

IMGSZ_BY_TIER = {
    "HIGH": 640,
    "MEDIUM": 480,
    "LOW": 320,
}

CONF = 0.25
IOU = 0.7
CAM_INDEX = 0

HIGH_ENTER = 70.0   
HIGH_EXIT  = 60.0   
LOW_ENTER  = 25.0   
LOW_EXIT   = 35.0   

TOP_K = 3
TAU = 0.78  

ALLOW_ESCALATION_IN_MEDIUM = True
ALLOW_ESCALATION_IN_LOW = False

LOG_PATH = "inference_log.csv"



@dataclass
class InferenceDecision:
    battery_pct: float
    state_before: str
    state_after: str
    base_tier: str
    final_tier: str
    escalated: bool
    frame_confidence: float
    infected_count: int
    latency_ms: float



def get_battery_percent() -> float:

    return 52.0


def update_battery_state(battery_pct: float, prev_state: str) -> str:
    if prev_state == "HIGH":
        if battery_pct < HIGH_EXIT:
            return "MEDIUM"
        return "HIGH"

    if prev_state == "MEDIUM":
        if battery_pct > HIGH_ENTER:
            return "HIGH"
        if battery_pct < LOW_ENTER:
            return "LOW"
        return "MEDIUM"

    if prev_state == "LOW":
        if battery_pct > LOW_EXIT:
            return "MEDIUM"
        return "LOW"

    return "MEDIUM"


def base_tier_for_state(state: str) -> str:
    if state == "HIGH":
        return "HIGH"
    if state == "MEDIUM":
        return "MEDIUM"
    return "LOW"


def get_infected_class_id(result, infected_name="infected"):
   
    names = result.names
    if isinstance(names, dict):
        for cls_id, cls_name in names.items():
            if str(cls_name).lower() == infected_name.lower():
                return int(cls_id)
    elif isinstance(names, list):
        for cls_id, cls_name in enumerate(names):
            if str(cls_name).lower() == infected_name.lower():
                return int(cls_id)
    raise ValueError(f"Could not find class '{infected_name}' in model names: {names}")


def extract_infected_confidences(result, infected_class_id):
    
    if result.boxes is None or len(result.boxes) == 0:
        return []

    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    infected_confs = confs[classes == infected_class_id]
    return infected_confs.tolist()


def compute_frame_confidence(infected_confidences, top_k=3):
    
    if len(infected_confidences) == 0:
        return 0.0

    infected_confidences = sorted(infected_confidences, reverse=True)
    top = infected_confidences[:min(top_k, len(infected_confidences))]
    return float(np.mean(top))



def next_higher_tier(current_tier: str):
    if current_tier == "LOW":
        return "MEDIUM"
    if current_tier == "MEDIUM":
        return "HIGH"
    return None


def escalation_allowed(state: str, current_tier: str) -> bool:
    if state == "HIGH":
        return False 
    if state == "MEDIUM":
        return current_tier == "MEDIUM" and ALLOW_ESCALATION_IN_MEDIUM
    if state == "LOW":
        return current_tier == "LOW" and ALLOW_ESCALATION_IN_LOW
    return False



class BatteryAwareMalariaController:
    def __init__(self, engine_paths, imgsz_by_tier, conf=0.25, iou=0.7, top_k=3, tau=0.78):
        self.models = {tier: YOLO(path) for tier, path in engine_paths.items()}
        self.imgsz_by_tier = imgsz_by_tier
        self.conf = conf
        self.iou = iou
        self.top_k = top_k
        self.tau = tau
        self.state = "MEDIUM"

        self._init_log()

    def _init_log(self):
        file_exists = Path(LOG_PATH).exists()
        self.log_file = open(LOG_PATH, "a", newline="")
        self.writer = csv.writer(self.log_file)
        if not file_exists:
            self.writer.writerow([
                "timestamp",
                "battery_pct",
                "state_before",
                "state_after",
                "base_tier",
                "final_tier",
                "escalated",
                "frame_confidence",
                "infected_count",
                "latency_ms"
            ])

    def _run_model(self, frame, tier):
        model = self.models[tier]
        imgsz = self.imgsz_by_tier[tier]

        t0 = time.perf_counter()
        result = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return result, latency_ms

    def infer(self, frame, battery_pct):
        state_before = self.state
        self.state = update_battery_state(battery_pct, self.state)
        state_after = self.state

        base_tier = base_tier_for_state(self.state)
        final_tier = base_tier
        escalated = False

        result, latency_ms = self._run_model(frame, base_tier)
        infected_class_id = get_infected_class_id(result, infected_name="infected")
        infected_confidences = extract_infected_confidences(result, infected_class_id)
        frame_conf = compute_frame_confidence(infected_confidences, top_k=self.top_k)
        infected_count = len(infected_confidences)

        if frame_conf < self.tau and escalation_allowed(self.state, base_tier):
            higher = next_higher_tier(base_tier)
            if higher is not None:
                result, latency_ms_2 = self._run_model(frame, higher)
                latency_ms += latency_ms_2
                final_tier = higher
                escalated = True

                infected_class_id = get_infected_class_id(result, infected_name="infected")
                infected_confidences = extract_infected_confidences(result, infected_class_id)
                frame_conf = compute_frame_confidence(infected_confidences, top_k=self.top_k)
                infected_count = len(infected_confidences)

        decision = InferenceDecision(
            battery_pct=battery_pct,
            state_before=state_before,
            state_after=state_after,
            base_tier=base_tier,
            final_tier=final_tier,
            escalated=escalated,
            frame_confidence=frame_conf,
            infected_count=infected_count,
            latency_ms=latency_ms
        )

        self.writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            round(decision.battery_pct, 2),
            decision.state_before,
            decision.state_after,
            decision.base_tier,
            decision.final_tier,
            int(decision.escalated),
            round(decision.frame_confidence, 4),
            decision.infected_count,
            round(decision.latency_ms, 2)
        ])
        self.log_file.flush()

        return result, decision

    def close(self):
        self.log_file.close()



def draw_overlay(frame, decision: InferenceDecision):
    lines = [
        f"Battery: {decision.battery_pct:.1f}%",
        f"State: {decision.state_after}",
        f"Base Tier: {decision.base_tier}",
        f"Final Tier: {decision.final_tier}",
        f"Escalated: {decision.escalated}",
        f"Frame Conf: {decision.frame_confidence:.3f}",
        f"Infected Count: {decision.infected_count}",
        f"Latency: {decision.latency_ms:.1f} ms",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 24
    return frame


def main():
    controller = BatteryAwareMalariaController(
        engine_paths=ENGINE_PATHS,
        imgsz_by_tier=IMGSZ_BY_TIER,
        conf=CONF,
        iou=IOU,
        top_k=TOP_K,
        tau=TAU
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            battery_pct = get_battery_percent()
            result, decision = controller.infer(frame, battery_pct)

            view = result.plot()
            view = draw_overlay(view, decision)

            cv2.imshow("Battery-Gated Malaria Detection", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        controller.close()


if __name__ == "__main__":
    main()

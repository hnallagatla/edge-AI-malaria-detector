# Resource-Aware Edge AI for Malaria Cell Detection

**An end-to-end malaria cell-screening prototype on the NVIDIA Jetson Orin Nano Super**  
*Independent research and engineering by Hemanth Nallagatla*

## Prototype Demonstration

**▶️ [Watch the working prototype demonstration on YouTube](https://youtu.be/SihIgRC8rqY)**

Demonstration of the Jetson Orin Nano Super prototype and its integrated camera-to-LCD interface.

**This project is described in my solo-authored paper:** **[Read my full IEEE BIBM 2026 research paper](BIBM_2026_UGHS_Hemanth_Nallagatla_Camera_Ready.pdf)**

> **Hardware and setup required:** The live prototype is **not a plug-and-play desktop application**. Running [`tools/lcd_display.py`](tools/lcd_display.py) requires a configured **NVIDIA Jetson Orin Nano Super** with a compatible JetPack/CUDA/TensorRT environment, a **USB webcam**, and a **connected ST7789 SPI LCD** with SPI enabled and the `jetson_orin_st7789` display driver installed. The relevant Python dependencies and model files must also be available. Some data-preparation and analysis scripts can run separately, but the integrated camera-to-LCD system cannot be reproduced by simply running the code on an ordinary laptop.

## The Project

How much computation is worth spending on a single image when a screening device has limited power?

I built an edge-AI system that detects infected and uninfected red blood cells in microscopy-style images, measures the tradeoffs among detection sensitivity, latency, and energy, and changes inference models when a frame warrants more computation. The prototype runs locally on an **NVIDIA Jetson Orin Nano Super**, with an LCD showing detections, cell counts, an estimated infected-cell fraction, the active inference tier, a magnified suspicious cell, and selective Eigen-CAM visualization.

The engineering challenge was not to invent a new neural-network architecture. It was to turn an existing detector into a **measured, resource-aware system**: create usable detection data, train and optimize models, evaluate them on the actual hardware, select defensible operating points, and integrate the pieces into a working device.

> **Research prototype, not a medical device.** This project has not been clinically validated and should not be used for diagnosis or treatment decisions.

## At a Glance

| | |
|---|---|
| **Hardware** | Jetson Orin Nano Super Developer Kit, USB webcam, ST7789 LCD |
| **Detector** | Fine-tuned YOLOv8s |
| **Optimization** | TensorRT FP16 and INT8 at 320, 480, 640, and 960 pixels |
| **Evaluation** | Infected-cell recall, precision, mAP50–95, inference latency, and whole-device energy |
| **Selected inference tiers** | INT8-320 (low energy), FP16-480 (balanced), FP16-640 (high sensitivity) |
| **Research artifacts** | [Models and experiments](experiments/) · [Evaluation results](evals/) · [Energy logs](tegrastats_logs/) · [Source code](tools/) |

## How I Built It

### 1. From cell crops to detection data

The public NIH/NLM malaria dataset contains images of individual infected and uninfected red blood cells. To build a multi-cell detector rather than a single-image classifier, I created a pipeline that removes image backgrounds, transforms cell cutouts, and places them into synthetic microscopy-style pages with automatically generated YOLO bounding-box labels.

The generator produced **2,000 training, 200 validation, and 200 test pages** at **1280 × 720**, with **25–55 cells per page**. I introduced changes in scale, rotation, lighting, blur, and compression, and wrote a separate tool to draw label previews so I could inspect the generated boxes. The common test set contained **7,469 labeled cells**.

**Explore:** [`precutout_rembg.py`](tools/precutout_rembg.py) · [`make_synth_pages.py`](tools/make_synth_pages.py) · [`preview_labels.py`](tools/preview_labels.py) · [`data/`](data/)

### 2. Train once; test many deployment configurations

I fine-tuned a COCO-pretrained **YOLOv8s** detector at 640-pixel input size. Training used automatic mixed precision and early stopping; the main run ended after **212 epochs** and retained the best checkpoint near epoch 162. Rather than assume that the highest-resolution model would be best on the Jetson, I exported that checkpoint to **eight TensorRT engines**: FP16 and post-training INT8 at four resolutions each.

All eight engines were evaluated against the same synthetic test pages. INT8 calibration used training images, not the test set. I focused on *infected-cell recall* because missed infected cells are especially relevant in a screening-oriented application.

**Explore:** [`experiments/y8s_fp32_640/`](experiments/y8s_fp32_640/) (checkpoints and exported engines) · [`evals/`](evals/) (evaluation artifacts)

### 3. Measure energy instead of guessing from model size

I used **TegraStats** to log the Jetson's `VDD_IN` power rail while configurations processed a common prerecorded video. I integrated power over the recorded time intervals and divided by completed inference count to obtain **whole-device energy per inference**. The LCD was off during these measurements, so these figures are for the compute platform—not the complete camera-and-display product.

Keeping latency, recall, and energy together made it possible to choose configurations based on measured tradeoffs rather than FP16/INT8 labels alone.

**Explore:** [`tegrastats_energy.py`](tools/tegrastats_energy.py) · [`tegrastats_logs/`](tegrastats_logs/) · [`tegrastats_*.txt`](tegrastats_fp16_30ips_640px.txt)

## What the Benchmarks Showed

These are the **final results reported in my IEEE BIBM 2026 UGHS paper**, measured on a common set of 200 synthetic test pages. Precision and recall refer to the **infected class**; latency refers to model inference, not the full camera-to-LCD loop.

| TensorRT configuration | Infected recall | mAP50–95 | Latency | Energy / inference |
|---|---:|---:|---:|---:|
| **INT8-320 — selected low-energy tier** | **94.3%** | 0.910 | **7.6 ms** | **0.198 J** |
| FP16-320 | 95.2% | 0.940 | 9.0 ms | 0.225 J |
| INT8-480 | 95.4% | 0.956 | 10.4 ms | 0.226 J |
| **FP16-480 — selected balanced tier** | **96.8%** | 0.974 | **11.4 ms** | **0.243 J** |
| INT8-640 | 96.9% | 0.964 | 10.6 ms | 0.256 J |
| **FP16-640 — selected high-sensitivity tier** | **97.5%** | 0.976 | **17.5 ms** | **0.282 J** |
| INT8-960 | 95.1% | 0.969 | 16.2 ms | 0.344 J |
| FP16-960 | 97.4% | 0.978 | 21.4 ms | 0.464 J |

The result that most shaped the build was that **more resolution was not always worth its cost**. FP16-960 consumed **64.5% more energy** than FP16-640 without improving infected-cell recall. Meanwhile, FP16-480 preserved recall within **0.7 percentage points** of FP16-640 while using **13.8% less energy** and **34.9% less inference time**.

That led to three operating roles rather than one supposedly “best” model: a low-energy starting point, a balanced middle tier, and a higher-sensitivity option when the system needs another look.

## From Measurements to a Working Device

The [integrated prototype](tools/lcd_display.py) begins with **INT8-320**. It combines infected-detection confidence with the top detection scores, smooths the result over time, and can escalate to **FP16-480** and then **FP16-640** when confidence falls below configured thresholds and the assumed battery state permits it.

```text
USB camera → INT8-320 → confidence check → sufficient → display
                        ↓
                   insufficient
                        ↓
                    FP16-480 → confidence check
                                     ↓
                                insufficient
                                     ↓
                                  FP16-640
                                     ↓
                        counts + zoom + optional Eigen-CAM
                                     ↓
                                  ST7789 LCD
```

Two interface decisions came from thinking about what a person reviewing a prediction would need:

- **Live suspicious-cell zoom** enlarges a low-confidence infected detection and briefly tracks it, instead of forcing the reviewer to locate a small box in a crowded frame.
- **Selective Eigen-CAM** shows a qualitative feature-response map inside predicted infected-cell boxes. Running it only when triggered limits its continuous processing overhead.

In one **43.70-second integration demonstration**, the application read **264 frames** (6.04 end-to-end frames/s), made **590 model invocations**, and recomputed Eigen-CAM on **11 frames**. This demonstrates that tier switching, visualization, and LCD rendering can operate together; it does not measure diagnostic accuracy on live microscope video.

**Implementation distinction:** The prototype currently uses a **simulated battery percentage**, not live battery telemetry. The separate [`confidence.py`](tools/confidence.py) file explores a different battery-state/hysteresis controller and is not the same control path as the LCD application. Neither controller has yet demonstrated net energy savings against fixed-tier baselines on labeled video.

## Finding Your Way Around the Repository

| Path | What's inside |
|---|---|
| [`data/`](data/) | Original cell images, cutouts, synthetic pages and YOLO labels, preview images, and prerecorded test input |
| [`experiments/`](experiments/) | Training artifacts, PyTorch checkpoints, ONNX exports, FP16/INT8 TensorRT engines, and exploratory YOLOv8n runs |
| [`evals/`](evals/) | Evaluation outputs across precision and resolution settings |
| [`runs/detect/`](runs/detect/) | Retained validation and other run artifacts; bulky generated prediction videos were omitted |
| [`tools/`](tools/) | Data generation, annotation preview, inference controllers, LCD/OLED applications, plotting, and energy analysis |
| [`tegrastats_logs/`](tegrastats_logs/) | Raw Jetson power measurements |
| [`inference_log.csv`](inference_log.csv) | Logged decisions from a controller experiment |

The main working-device application is **[`tools/lcd_display.py`](tools/lcd_display.py)**. The root `main.py` is not the deployment entry point. Some historical plots and scripts preserve exploratory results; use the **final benchmark table above and the paper** for reported performance.

### Reproduction notes

Model weights and dataset images are stored with **Git LFS**. To retrieve them after cloning, install Git LFS and run `git lfs pull`. TensorRT `.engine` files depend on the compatible Jetson software and hardware environment; running the LCD application also requires the display driver, camera, and other device-specific dependencies. The repository preserves research files but is **not** a tested one-command installer.

## What Still Needs Validation

The synthetic pages were generated separately, but **underlying source-cell images were not split by patient or slide before synthesis**. Some augmented cell instances could therefore appear in both training and testing. The 97.5% infected-cell recall is **not** a claim about new patients, laboratories, or clinical sensitivity. The energy results also came from **one trial per configuration**, with the LCD disabled.

The next engineering steps are to test genuinely held-out patient/slide data, validate microscope acquisition under controlled optics and staining conditions, connect real battery telemetry, repeat full-system energy trials, and compare the adaptive cascade against fixed-tier inference on labeled video. Eigen-CAM is a visualization aid, not a causal medical explanation; the displayed infected-cell fraction is not a validated clinical parasitemia measurement.

## Research and Credits

This project is described in my solo-authored paper: **[Read my full IEEE BIBM 2026 research paper](BIBM_2026_UGHS_Hemanth_Nallagatla_Camera_Ready.pdf)**

**Hemanth Nallagatla**, *“Resource-Aware Edge AI for Malaria Cell Detection: Precision-Resolution Tradeoffs and Tiered TensorRT Inference.”* Accepted for presentation and publication at the **IEEE BIBM 2026 Undergraduate and High School Symposium (UGHS)**.

Source cell images: **NIH/NLM malaria thin-blood-smear dataset**, described by **Rajaraman et al.**, “Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images,” *PeerJ* 6:e4568 (2018), DOI: [10.7717/peerj.4568](https://doi.org/10.7717/peerj.4568). The project uses third-party tools including Ultralytics YOLO, NVIDIA TensorRT/TegraStats, PyTorch, and open-source display libraries; their respective license terms apply. Dataset redistribution rights should be checked independently of repository visibility.

**This repository documents a systems research prototype, not a clinical diagnostic product.**

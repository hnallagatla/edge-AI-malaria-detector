import os
import random
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm



def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def trim_transparent(img_rgba: Image.Image, min_size=8) -> Image.Image:
    
    if img_rgba.mode != "RGBA":
        img_rgba = img_rgba.convert("RGBA")

    alpha = img_rgba.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return img_rgba

    cropped = img_rgba.crop(bbox)
    if cropped.width < min_size or cropped.height < min_size:
        return img_rgba
    return cropped


def add_alpha_if_missing(cell_img: Image.Image, white_thresh=245) -> Image.Image:
    
    if cell_img.mode == "RGBA":
        return cell_img

    rgb = np.array(cell_img.convert("RGB"))
    mask_bg = (rgb[:, :, 0] >= white_thresh) & (rgb[:, :, 1] >= white_thresh) & (rgb[:, :, 2] >= white_thresh)
    alpha = np.where(mask_bg, 0, 255).astype(np.uint8)

    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.8)

    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def make_paper_background(w: int, h: int) -> Image.Image:
    base = np.full((h, w, 3), 245, dtype=np.uint8)

    noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = r / r.max()
    vignette = (1.0 - 0.06 * (r ** 1.7)) 
    base = np.clip(base.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

    return Image.fromarray(base, mode="RGB")


def add_global_effects(canvas: Image.Image, add_blur=True):
    canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.90, 1.12))
    canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.90, 1.18))

    if random.random() < 0.35:
        canvas = ImageEnhance.Color(canvas).enhance(random.uniform(0.92, 1.10))

    bgr = pil_to_bgr(canvas)

    if add_blur and random.random() < 0.65:
        k = random.choice([3, 5])
        bgr = cv2.GaussianBlur(bgr, (k, k), sigmaX=random.uniform(0.2, 1.0))

    if random.random() < 0.65:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 95)]
        ok, enc = cv2.imencode(".jpg", bgr, encode_param)
        if ok:
            bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return bgr_to_pil(bgr)



def random_cell_transform(cell: Image.Image, scale_range=(0.6, 1.4), rot_deg=15, ensure_alpha=True):

    if ensure_alpha:
        cell = add_alpha_if_missing(cell)
    else:
        cell = cell.convert("RGBA")

    s = random.uniform(*scale_range)
    new_w = max(12, int(cell.width * s))
    new_h = max(12, int(cell.height * s))
    cell = cell.resize((new_w, new_h), resample=Image.BICUBIC)

    ang = random.uniform(-rot_deg, rot_deg)
    cell = cell.rotate(ang, resample=Image.BICUBIC, expand=True)

    cell = trim_transparent(cell, min_size=10)

    if random.random() < 0.85:
        rgb = cell.convert("RGB")
        rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.85, 1.18))
        rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.85, 1.22))
        a = cell.split()[-1]
        cell = Image.merge("RGBA", (*rgb.split(), a))

    if random.random() < 0.25:
        cell = cell.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    return cell


def boxes_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def save_yolo_labels(label_path: Path, boxes_xyxy, classes, img_w, img_h):
    lines = []
    for (x1, y1, x2, y2), cls in zip(boxes_xyxy, classes):
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        cx_n = clamp(cx / img_w, 0.0, 1.0)
        cy_n = clamp(cy / img_h, 0.0, 1.0)
        bw_n = clamp(bw / img_w, 0.0, 1.0)
        bh_n = clamp(bh / img_h, 0.0, 1.0)

        if bw_n <= 0 or bh_n <= 0:
            continue

        lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_yolo_data_yaml(out_dir: Path):

    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
        f"  0: uninfected\n"
        f"  1: infected\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text)


def generate_dataset(
    infected_dir: Path,
    uninfected_dir: Path,
    out_dir: Path,
    img_size=(1280, 720),
    train_count=2000,
    val_count=200,
    test_count=200,
    cells_per_page=(25, 55),
    max_iou=0.03,
    seed=42,
    ensure_alpha=True,
):
    random.seed(seed)
    np.random.seed(seed)

    inf = list_images(infected_dir)
    uninf = list_images(uninfected_dir)
    if len(inf) == 0 or len(uninf) == 0:
        raise RuntimeError("No images found. Check your infected/uninfected folders.")

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def make_split(split_name: str, count: int):
        w, h = img_size

        for i in tqdm(range(count), desc=f"Generating {split_name}"):
            canvas = make_paper_background(w, h).convert("RGBA")

            boxes = []
            classes = []
            n_cells = random.randint(*cells_per_page)

            tries_total = 0
            placed = 0
            max_tries = n_cells * 35  
            while placed < n_cells and tries_total < max_tries:
                tries_total += 1

                cls = 1 if random.random() < 0.5 else 0 
                src = random.choice(inf if cls == 1 else uninf)

                cell = Image.open(src)
                cell_t = random_cell_transform(cell, ensure_alpha=ensure_alpha)

                if cell_t.width >= w - 4 or cell_t.height >= h - 4:
                    continue

                x1 = random.randint(0, w - cell_t.width)
                y1 = random.randint(0, h - cell_t.height)
                x2 = x1 + cell_t.width
                y2 = y1 + cell_t.height

                new_box = (x1, y1, x2, y2)

                ok = True
                for b in boxes:
                    if boxes_iou(new_box, b) > max_iou:
                        ok = False
                        break
                if not ok:
                    continue

                canvas.alpha_composite(cell_t, (x1, y1))

                boxes.append(new_box)
                classes.append(cls)
                placed += 1

            canvas_rgb = canvas.convert("RGB")
            canvas_rgb = add_global_effects(canvas_rgb, add_blur=True)

            stem = f"page_{split_name}_{i:06d}"
            img_path = out_dir / "images" / split_name / f"{stem}.jpg"
            label_path = out_dir / "labels" / split_name / f"{stem}.txt"

            canvas_rgb.save(img_path, quality=92)
            save_yolo_labels(label_path, boxes, classes, w, h)

    make_split("train", train_count)
    make_split("val", val_count)
    make_split("test", test_count)

    write_yolo_data_yaml(out_dir)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    infected = project_root / "data" / "nih_cutout" / "infected"
    uninfected = project_root / "data" / "nih_cutout" / "uninfected"

    out = project_root / "data" / "synth_det_pages"

    generate_dataset(
        infected_dir=infected,
        uninfected_dir=uninfected,
        out_dir=out,
        img_size=(1280, 720),
        train_count=2000,
        val_count=200,
        test_count=200,
        cells_per_page=(25, 55),
        max_iou=0.03,
        seed=42,
        ensure_alpha=True,  
    )

    print(f"Done. Wrote synthetic YOLO detection dataset to: {out}")
    print(f"YOLO config written to: {out / 'data.yaml'}")

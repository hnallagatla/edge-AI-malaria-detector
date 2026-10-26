from pathlib import Path
import random
import cv2

NAMES = {0: "uninfected", 1: "infected"}

def draw_one(img_path: Path, label_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read {img_path}")

    h, w = img.shape[:2]

    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(0, min(w-1, x2))
            y2 = max(0, min(h-1, y2))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, NAMES.get(cls, str(cls)), (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = root / "data" / "synth_det_pages"  # change if needed

    split = "train"
    imgs = sorted((data / "images" / split).glob("*.jpg"))
    random.shuffle(imgs)

    out_dir = root / "data" / "label_previews" / split
    for p in imgs[:20]:
        lp = data / "labels" / split / (p.stem + ".txt")
        out = out_dir / (p.stem + "_preview.jpg")
        draw_one(p, lp, out)

    print("Wrote previews to:", out_dir)
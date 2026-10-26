from pathlib import Path
from tqdm import tqdm
from PIL import Image
from rembg import remove, new_session

def list_imgs(folder: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

def convert_folder(in_dir: Path, out_dir: Path, session):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_imgs(in_dir)
    for p in tqdm(files, desc=f"cutout {in_dir.name}"):
        img = Image.open(p).convert("RGBA")
        out = remove(img, session=session) 
        out.save(out_dir / (p.stem + ".png"))

if __name__ == "__main__":
    session = new_session(providers=["CPUExecutionProvider"]) 

    project_root = Path(__file__).resolve().parents[1]
    inf_in  = project_root / "data" / "nih_base" / "infected"
    un_in   = project_root / "data" / "nih_base" / "uninfected"

    inf_out = project_root / "data" / "nih_cutout" / "infected"
    un_out  = project_root / "data" / "nih_cutout" / "uninfected"

    convert_folder(inf_in, inf_out, session)
    convert_folder(un_in, un_out, session)

    print("Done. Wrote cutouts to data/nih_cutout/")

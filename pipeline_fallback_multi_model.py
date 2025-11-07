# pipeline_yolo_patch_cal8bit.py
# 1) susun dataset  2) auto-label  3) ship.yaml  4) train YOLO (v12→v11)

CONFIG = {
    "DATA_ROOT": r"C:\PUNYA_HELENA2\OPENSARSHIP_1\PATCH_CAL_8bit",  # hasil konversi
    "OUT_ROOT":  "datasets",

    "DO_PREPARE":   True,
    "DO_AUTOLABEL": True,
    "DO_YAML":      True,
    "DO_TRAIN":     True,   # nyalakan training

    # Auto-label (CFAR) — kalau terlalu banyak/sedikit deteksi, atur K & WIN
    "K": 3.0,        # 2.5–3.5 (lebih kecil → lebih sensitif)
    "WIN": 31,       # 21/31/41

    # Training
    "MODEL_TRY": ["yolov12n.pt", "yolov11n.pt"],  # coba v12 dulu
    "IMGSZ": 512,   # bisa 640 nanti
    "EPOCHS": 20,   # bisa dinaikkan nanti
    "BATCH": 8,
    "YAML_OUT": "ship.yaml",
}

# ----------------- implementasi -----------------
import os, glob, random, shutil, yaml
import numpy as np
import cv2

def is_img(p): return p.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))
def ensure(p): os.makedirs(p, exist_ok=True)

def prepare_dataset(data_root, out_root, train=0.7, val=0.2, test=0.1, seed=42):
    random.seed(seed)
    files = [p for p in glob.glob(os.path.join(data_root, "**", "*.*"), recursive=True) if is_img(p)]
    assert files, f"Tidak ada gambar di {data_root}"
    for sp in ["train","val","test"]:
        ensure(os.path.join(out_root, "images", sp))
        ensure(os.path.join(out_root, "labels", sp))
    random.shuffle(files); n=len(files)
    n_tr=int(n*train); n_va=int(n*val)
    splits = {"train": files[:n_tr], "val": files[n_tr:n_tr+n_va], "test": files[n_tr+n_va:]}
    for sp, paths in splits.items():
        for src in paths:
            dst = os.path.join(out_root, "images", sp, os.path.basename(src))
            shutil.copy2(src, dst)
    print(f"[prepare] {n} gambar → {out_root}/images/train|val|test")

def cfar_mask(gray, k=3.0, win=31):
    if gray.ndim==3 and gray.shape[2]>1:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (win,win), normalize=True)
    sqm  = cv2.boxFilter(f*f, -1, (win,win), normalize=True)
    var  = np.clip(sqm - mean*mean, 0, None)
    std  = np.sqrt(var + 1e-6)
    thr  = mean + k*std
    return (f > thr).astype(np.uint8)*255

def biggest_box(mask):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x,y,w,h

def auto_label(out_root, k=3.0, win=31):
    for sp in ["train","val","test"]:
        imgd = os.path.join(out_root, "images", sp)
        lbld = os.path.join(out_root, "labels", sp); ensure(lbld)
        prev = os.path.join(out_root, "preview_masks", sp); ensure(prev)
        for ip in glob.glob(os.path.join(imgd, "*")):
            if not is_img(ip): continue
            im = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
            if im is None: continue
            if im.ndim==3 and im.shape[2]>1:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            m  = cfar_mask(im, k=k, win=win)
            m  = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            box = biggest_box(m)
            h,w = im.shape[:2]
            txt = os.path.join(lbld, os.path.splitext(os.path.basename(ip))[0] + ".txt")
            if box is None:
                open(txt,"w").close()
            else:
                x,y,bw,bh = box
                xc=(x+bw/2)/w; yc=(y+bh/2)/h; nw=bw/w; nh=bh/h
                with open(txt,"w") as f:
                    f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
            cv2.imwrite(os.path.join(prev, os.path.basename(ip)), m)
        print(f"[autolabel] {sp} ✔")

def write_yaml(out_root, out_yaml):
    data = {"path":".",
            "train":f"{out_root}/images/train",
            "val":  f"{out_root}/images/val",
            "test": f"{out_root}/images/test",
            "names": {0: "ship"}}
    with open(out_yaml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[yaml] Ditulis: {out_yaml}")

def train(model_list, yaml_path, imgsz, epochs, batch):
    try:
        from ultralytics import YOLO
    except Exception:
        print("[train] ultralytics belum terpasang. Jalankan: pip install ultralytics")
        return
    last=None
    for m in model_list:
        try:
            print(f"[train] coba: {m}")
            YOLO(m).train(data=yaml_path, imgsz=imgsz, epochs=epochs, batch=batch, task="detect")
            print("[train] selesai. Lihat runs/detect/train/weights/best.pt")
            return
        except Exception as e:
            print(f"[train] gagal {m}: {e}"); last=e
    raise RuntimeError(f"Semua model gagal. Error terakhir: {last}")

# ----------------- jalankan -----------------
cfg = CONFIG
if cfg["DO_PREPARE"]:
    prepare_dataset(cfg["DATA_ROOT"], cfg["OUT_ROOT"])
if cfg["DO_AUTOLABEL"]:
    auto_label(cfg["OUT_ROOT"], k=cfg["K"], win=cfg["WIN"])
if cfg["DO_YAML"]:
    write_yaml(cfg["OUT_ROOT"], cfg["YAML_OUT"])
if cfg["DO_TRAIN"]:
    train(cfg["MODEL_TRY"], cfg["YAML_OUT"], cfg["IMGSZ"], cfg["EPOCHS"], cfg["BATCH"])

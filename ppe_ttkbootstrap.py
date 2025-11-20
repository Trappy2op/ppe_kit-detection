"""
ppe_ttkbootstrap_full.py

Full PPE detection GUI using ttkbootstrap + YOLOv8n.
Supports: Upload Image, Upload Video, Webcam, Drag & Drop (optional)
Saves snapshots and annotated outputs to ./gui_outputs/
Rule-based PPE violation detection integrated:
 - Detects persons & PPE boxes
 - Marks person unsafe if helmet or vest missing (adjustable)
 - Draws red box + missing-items label on unsafe persons
"""

import os
import time
import threading
import traceback
from pathlib import Path
from queue import Queue, Empty

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# Optional drag & drop
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES  # type: ignore
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_USER = PROJECT_ROOT / "models" / "best.pt"   # your model (optional)
DEFAULT_MODEL = "yolov8n.pt"                      # fallback - Ultralytics will download if needed
OUTPUT_DIR = PROJECT_ROOT / "gui_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_DEFAULT = 0.40
CONF_MIN, CONF_MAX = 0.10, 0.90

# classes that explicitly indicate 'NO-' in dataset (we still handle rule-based missing-PPE if NO- classes are absent)
UNSAFE_LABELS = {
    "NO-Hardhat", "NO-Helmet", "NO-Mask", "NO-Vest",
    "NO-Safety Vest", "NO-Safety Shoes", "NO-Goggles", "NO-Gloves"
}
# ----------------------------------------

# Choose model path
MODEL_PATH = str(MODEL_USER) if MODEL_USER.exists() else DEFAULT_MODEL

# Load model once (may take a moment)
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    model = None
    print("Could not load model:", e)
    traceback.print_exc()

# ---------- Helpers ----------
def bgr_to_tk(frame_bgr, max_w=960, max_h=540):
    """Convert BGR numpy image to Tk PhotoImage scaled to fit box."""
    try:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        img_rgb = frame_bgr
    pil = Image.fromarray(img_rgb)
    w, h = pil.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale != 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), Image.ANTIALIAS)
    return ImageTk.PhotoImage(pil)

def center_of_box(box):
    """Return center (x,y) of xyxy box."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def box_inside(box_inner, box_outer):
    """Return True if box_inner is fully inside box_outer (xyxy)."""
    x1i, y1i, x2i, y2i = box_inner
    x1o, y1o, x2o, y2o = box_outer
    return (x1i >= x1o) and (y1i >= y1o) and (x2i <= x2o) and (y2i <= y2o)

def any_substr_in(text, keywords):
    """Case-insensitive substring match for any keyword in text."""
    t = text.lower()
    for k in keywords:
        if k.lower() in t:
            return True
    return False

# PPE detection keywords (robust to label variations)
HELMET_KEYS = ("hardhat", "helmet", "hat")
VEST_KEYS = ("vest", "safety vest", "safety_vest")
MASK_KEYS = ("mask",)
SHOE_KEYS = ("shoe", "safety_shoe", "safety_shoes")
GOGGLE_KEYS = ("goggle", "goggles")
GLOVE_KEYS = ("glove", "gloves")

REQUIRED_FOR_SAFE = ("helmet", "vest")  # conservative: require helmet AND vest to be considered safe; change as desired

def analyze_and_annotate(results, frame, conf_threshold=0.3):
    """
    Analyze YOLO results and annotate the frame with:
     - person boxes: green if safe, red if unsafe (with missing items)
     - other PPE boxes as cyan (optional)
    Returns:
      annotated_frame, safety_pct, total_persons, unsafe_persons
    """
    annotated = frame.copy()
    if not results or len(results) == 0:
        return annotated, 100.0, 0, 0

    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return annotated, 100.0, 0, 0

    # extract xyxy and class ids and names
    try:
        xyxy = boxes.xyxy.cpu().numpy()  # shape (N,4)
        cls_ids = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        try:
            xyxy = np.array([b.xyxy[0].cpu().numpy() for b in boxes])
            cls_ids = np.array([int(b.cls[0].cpu().numpy()) for b in boxes])
        except Exception:
            return annotated, 100.0, 0, 0

    names_map = r.names

    persons = []   # list of (box, cls_id, idx)
    ppe_items = [] # list of (box, label, idx)

    for idx, (box, cid) in enumerate(zip(xyxy, cls_ids)):
        label = names_map.get(int(cid), str(cid))
        # filter low-confidence boxes? ultralytics already filtered by conf param
        # classify into person vs ppe
        if label.lower() == "person":
            persons.append((box, label, idx))
        else:
            ppe_items.append((box, label, idx))

    total_persons = len(persons)
    unsafe_persons = 0

    # For each person, check ppe presence by checking PPE box centers inside person's bbox
    person_annotations = []  # list of dicts with keys: box, safe(bool), missing(list)
    for p_box, p_label, p_idx in persons:
        found = {
            "helmet": False,
            "vest": False,
            "mask": False,
            "shoe": False,
            "goggle": False,
            "glove": False
        }
        # Check NO- classes first (explicit unsafe detections)
        # If any NO-* box lies inside the person -> count as unsafe and list missing item
        explicit_no_missing = []
        for b_box, b_label, b_idx in ppe_items:
            # check if this item is a NO- type
            if isinstance(b_label, str) and b_label.upper().startswith("NO-"):
                # if NO- box inside person bbox -> count corresponding missing
                if box_inside(b_box, p_box):
                    explicit_no_missing.append(b_label[3:])  # text after NO-
        # Now detect positive PPE items
        for b_box, b_label, b_idx in ppe_items:
            lb = str(b_label).lower()
            # center inside person
            cx, cy = center_of_box(b_box)
            if (cx >= p_box[0] and cy >= p_box[1] and cx <= p_box[2] and cy <= p_box[3]):
                if any_substr_in(lb, HELMET_KEYS):
                    found["helmet"] = True
                if any_substr_in(lb, VEST_KEYS):
                    found["vest"] = True
                if any_substr_in(lb, MASK_KEYS):
                    found["mask"] = True
                if any_substr_in(lb, SHOE_KEYS):
                    found["shoe"] = True
                if any_substr_in(lb, GOGGLE_KEYS):
                    found["goggle"] = True
                if any_substr_in(lb, GLOVE_KEYS):
                    found["glove"] = True

        # compute missing list:
        missing = []
        # If explicit NO-* entries were found, trust them (they indicate missing items)
        if explicit_no_missing:
            # normalize explicit names (strip/uppercase)
            for ex in explicit_no_missing:
                missing.append(ex.strip())
        else:
            # rule-based missing detection:
            # We treat 'helmet' and 'vest' as required items (conservative)
            if not found["helmet"]:
                missing.append("helmet")
            if not found["vest"]:
                missing.append("vest")
            # Optional: comment/uncomment to add more strict requirements:
            # if not found["mask"]:
            #     missing.append("mask")
            # if not found["shoe"]:
            #     missing.append("shoe")

        is_safe = (len(missing) == 0)
        if not is_safe:
            unsafe_persons += 1

        person_annotations.append({
            "box": p_box.astype(int),
            "missing": missing,
            "safe": is_safe
        })

    # Draw annotations
    # First draw PPE / other detections lightly (optional)
    for b_box, b_label, idx in ppe_items:
        x1, y1, x2, y2 = b_box.astype(int)
        # cyan for PPE items
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 220, 255), 2)
        txt = f"{b_label}"
        cv2.putText(annotated, txt, (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 2)

    # Draw persons with color depending on safety
    for pa in person_annotations:
        x1, y1, x2, y2 = pa["box"]
        if pa["safe"]:
            color = (50, 200, 50)  # green
            label = "Safe"
        else:
            color = (0, 0, 255)    # red
            missing_txt = ", ".join(pa["missing"])
            label = f"UNSAFE: {missing_txt}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Compute safety %
    safe_persons = max(total_persons - unsafe_persons, 0)
    safety_pct = 100.0 if total_persons == 0 else (safe_persons / total_persons) * 100.0

    # Big overlay text
    overlay_text = f"Safety: {safety_pct:.1f}%"
    overlay_color = (0, 200, 0) if safety_pct >= 70 else (0, 200, 200) if safety_pct >= 40 else (0, 0, 255)
    cv2.putText(annotated, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, overlay_color, 3)

    return annotated, safety_pct, total_persons, unsafe_persons

def save_snapshot_image(img_bgr, prefix="snapshot"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.jpg"
    path = OUTPUT_DIR / fname
    cv2.imwrite(str(path), img_bgr)
    return path

# ---------- Worker ----------
class InferenceWorker(threading.Thread):
    """Inference worker for video/webcam source."""
    def __init__(self, source, conf, frame_queue, stats_queue, stop_event):
        super().__init__(daemon=True)
        self.source = source
        self.conf = conf
        self.frame_queue = frame_queue
        self.stats_queue = stats_queue
        self.stop_event = stop_event
        self.cap = None

    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                # try numeric index
                try:
                    idx = int(self.source)
                    self.cap = cv2.VideoCapture(idx)
                except Exception:
                    pass
            if not self.cap.isOpened():
                self.stats_queue.put(("error", f"Cannot open source: {self.source}"))
                return
        except Exception as e:
            self.stats_queue.put(("error", f"OpenCV error: {e}"))
            return

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                results = model(frame, conf=self.conf)
            except Exception as e:
                self.stats_queue.put(("error", f"Inference error: {e}"))
                results = None

            if results:
                annotated, safety, tot, unsafe = analyze_and_annotate(results, frame, conf_threshold=self.conf)
            else:
                annotated = frame
                safety, tot, unsafe = 100.0, 0, 0

            # keep only latest frame in queue (drop old)
            try:
                while True:
                    self.frame_queue.get_nowait()
            except Empty:
                pass
            try:
                self.frame_queue.put_nowait(annotated)
            except Exception:
                pass
            self.stats_queue.put(("stats", (safety, tot, unsafe)))
            time.sleep(0.01)

        if self.cap:
            self.cap.release()

# ---------- GUI ----------
class PPEApp:
    def __init__(self, root):
        # If tkinterdnd2 is available, root might be a TkinterDnD.Tk instance
        self.root = root
        self.root.title("PPE Detection — ttkbootstrap")
        self.root.geometry("1200x720")
        self.style = tb.Style(theme="flatly")

        # main frames
        self.left = tb.Frame(self.root)
        self.left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.right = tb.Frame(self.root, width=320)
        self.right.pack(side="right", fill="y", padx=6, pady=6)

        # preview label (create once)
        self.preview_label = tb.Label(self.left, text="No feed", anchor="center", background="black")
        self.preview_label.pack(fill="both", expand=True)

        # control widgets
        tb.Label(self.right, text="Controls", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(2,6))
        self.conf_var = tb.DoubleVar(value=CONF_DEFAULT)
        tb.Label(self.right, text="Confidence").pack(anchor="w")
        self.conf_scale = tb.Scale(self.right, from_=CONF_MIN, to=CONF_MAX, orient="horizontal", variable=self.conf_var, bootstyle="info")
        self.conf_scale.pack(fill="x", padx=6, pady=(0,8))
        self.conf_val = tb.Label(self.right, text=f"{CONF_DEFAULT:.2f}")
        self.conf_val.pack(anchor="e", padx=6)
        self.conf_var.trace_add("write", self._on_conf_change)

        btn_frame = tb.Frame(self.right)
        btn_frame.pack(fill="x", pady=8)
        tb.Button(btn_frame, text="Upload Image", bootstyle="primary", command=self.upload_image).pack(fill="x", pady=4)
        tb.Button(btn_frame, text="Upload Video", bootstyle="secondary", command=self.upload_video).pack(fill="x", pady=4)
        tb.Button(btn_frame, text="Start Webcam", bootstyle="success", command=self.start_webcam).pack(fill="x", pady=4)
        tb.Button(btn_frame, text="Stop", bootstyle="danger", command=self.stop_worker).pack(fill="x", pady=4)

        # drag & drop label
        if DND_AVAILABLE:
            tb.Label(self.right, text="Drag & Drop files onto preview (images/videos)").pack(anchor="w", padx=6, pady=(8,2))
            self.preview_label.drop_target_register(DND_FILES)  # type: ignore
            self.preview_label.dnd_bind('<<Drop>>', self._on_drop)  # type: ignore
        else:
            tb.Label(self.right, text="(Install tkinterdnd2 for drag & drop)", foreground="gray").pack(anchor="w", padx=6, pady=(8,2))

        tb.Button(self.right, text="Snapshot", command=self.take_snapshot).pack(fill="x", pady=6)
        tb.Button(self.right, text="Open output folder", command=lambda: os.startfile(str(OUTPUT_DIR))).pack(fill="x", pady=4)

        # stats
        tb.Label(self.right, text="Live Stats", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(10,0))
        self.stats_safety = tb.Label(self.right, text="Safety: N/A", font=("Helvetica", 10))
        self.stats_safety.pack(anchor="w", padx=6, pady=(4,2))
        self.stats_people = tb.Label(self.right, text="People: 0", font=("Helvetica", 10))
        self.stats_people.pack(anchor="w", padx=6)
        self.stats_unsafe = tb.Label(self.right, text="Unsafe: 0", font=("Helvetica", 10))
        self.stats_unsafe.pack(anchor="w", padx=6, pady=(0,8))

        # status bar
        self.status = tb.StringVar(value="Ready")
        tb.Label(self.root, textvariable=self.status, relief="sunken", anchor="w").pack(fill="x", side="bottom")

        # internal
        self.frame_queue = Queue(maxsize=1)
        self.stats_queue = Queue()
        self.worker = None
        self.worker_stop_event = None
        self.current_frame = None
        self.preview_photo = None

        # start periodic update
        self.root.after(50, self._periodic_update)

    def _on_conf_change(self, *a):
        self.conf_val.config(text=f"{self.conf_var.get():.2f}")

    # ---------- actions ----------
    def upload_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self.stop_worker()
        self.status.set(f"Detecting image: {os.path.basename(path)}")
        self._run_single_image(path)

    def upload_video(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Select video", filetypes=[("Videos", "*.mp4 *.avi *.mov")])
        if not path:
            return
        self.stop_worker()
        self.status.set(f"Playing video: {os.path.basename(path)}")
        self._start_worker(source=path)

    def start_webcam(self):
        self.stop_worker()
        self.status.set("Opening webcam...")
        self._start_worker(source=0)

    def stop_worker(self):
        if self.worker and self.worker.is_alive():
            self.status.set("Stopping...")
            self.worker_stop_event.set()
            self.worker.join(timeout=2)
        self.worker = None
        self.worker_stop_event = None
        self.status.set("Stopped")

    def take_snapshot(self):
        if self.current_frame is None:
            tb.messagebox.showinfo("No frame", "No frame available to save.")
            return
        out = save_snapshot_image(self.current_frame, prefix="evidence")
        self.status.set(f"Snapshot saved: {out}")
        tb.messagebox.showinfo("Saved", f"Snapshot saved:\n{out}")

    def _on_drop(self, event):
        # event.data could be a list-like string of paths
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if not paths:
            return
        first = paths[0]
        ext = first.split(".")[-1].lower()
        if ext in ("jpg", "jpeg", "png"):
            self.stop_worker()
            self._run_single_image(first)
        elif ext in ("mp4", "avi", "mov"):
            self.stop_worker()
            self._start_worker(source=first)
        else:
            tb.messagebox.showinfo("Unsupported", "Only images/videos supported.")

    def _run_single_image(self, path):
        try:
            # run inference on image (Ultralytics accepts file path)
            results = model(path, conf=float(self.conf_var.get()))
        except Exception as e:
            tb.messagebox.showerror("Inference error", str(e))
            return

        # load original image for robust drawing (use cv2 to read exact pixels)
        img = cv2.imread(path)
        if img is None:
            tb.messagebox.showerror("Error", f"Cannot load image: {path}")
            return

        annotated, safety, tot, unsafe = analyze_and_annotate(results, img, conf_threshold=float(self.conf_var.get()))
        self._display_frame(annotated)
        self._update_stats_ui(safety, tot, unsafe)
        out = OUTPUT_DIR / f"image_{int(time.time())}.jpg"
        cv2.imwrite(str(out), annotated)
        self.status.set(f"Saved annotated image: {out}")

    def _start_worker(self, source=0):
        self.worker_stop_event = threading.Event()
        self.worker = InferenceWorker(source, float(self.conf_var.get()), self.frame_queue, self.stats_queue, self.worker_stop_event)
        self.worker.start()
        self.status.set("Worker running...")

    # ---------- GUI update loop ----------
    def _periodic_update(self):
        # get latest frame
        try:
            frame = self.frame_queue.get_nowait()
            self._display_frame(frame)
        except Empty:
            pass

        # get stats
        try:
            while True:
                item = self.stats_queue.get_nowait()
                if item[0] == "stats":
                    safety, tot, unsafe = item[1]
                    self._update_stats_ui(safety, tot, unsafe)
                elif item[0] == "error":
                    self.status.set(item[1])
                    tb.messagebox.showerror("Error", item[1])
        except Empty:
            pass

        self.root.after(50, self._periodic_update)

    # ---------- display helper (safe) ----------
    def _display_frame(self, frame_bgr):
        try:
            self.current_frame = frame_bgr.copy()
            photo = bgr_to_tk(frame_bgr, max_w=960, max_h=540)

            # keep strong references to avoid GC
            self.preview_photo = photo
            self.preview_label.image = photo

            # update when idle to avoid race conditions
            self.preview_label.after_idle(lambda p=photo: self.preview_label.config(image=p, text=""))

        except Exception as e:
            print("Display error:", e)

    def _update_stats_ui(self, safety, tot, unsafe):
        self.stats_safety.config(text=f"Safety: {safety:.1f}%")
        self.stats_people.config(text=f"People: {tot}")
        self.stats_unsafe.config(text=f"Unsafe: {unsafe}")
        # color
        if safety >= 70:
            fg = "#2ecc71"
        elif safety >= 40:
            fg = "#f1c40f"
        else:
            fg = "#e74c3c"
        self.stats_safety.config(foreground=fg)
        self.status.set(f"Safety {safety:.1f}% — People {tot} — Unsafe {unsafe}")

# ---------- entrypoint ----------
def main():
    # If tkinterdnd2 available, create its Tk root else use ttkbootstrap Window
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()  # type: ignore
        # ttkbootstrap styling attaches to this root below
        app_root = root
    else:
        app_root = tb.Window(themename="flatly")

    app = PPEApp(app_root)
    app_root.mainloop()

if __name__ == "__main__":
    main()

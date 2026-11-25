"""Small image helpers for the snapshot scripts: crop a person, draw labelled boxes.

Works with OpenCV (cv2) OR Pillow (PIL), whichever is installed (cv2 is preferred).
Force one with the environment variable VLM_IMG_BACKEND=cv2 | pil.
No other dependency; nothing is ever written next to the original frames.
"""
import base64
import io
import os

BACKEND = None
_want = os.environ.get("VLM_IMG_BACKEND", "").lower()
if _want in ("", "cv2"):
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        BACKEND = "cv2"
    except ImportError:
        pass
if BACKEND is None and _want in ("", "pil"):
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        BACKEND = "pil"
    except ImportError:
        pass

# name -> RGB
PALETTE = {
    "ok": (40, 200, 60),         # green: color and type known
    "unknown": (255, 150, 0),    # orange: model said unknown for color or type
    "error": (230, 30, 30),      # red: request/parse failed for this person
    "skipped": (150, 150, 150),  # gray: box too small, not sent
}


def _need_backend():
    if BACKEND is None:
        raise RuntimeError("Neither OpenCV (cv2) nor Pillow (PIL) is installed in this Python. "
                           "Install one of them: pip install pillow   (or: pip install opencv-python-headless)")


def load_image(path: str):
    """Read an image file; returns a backend-specific object. Raises OSError if unreadable."""
    _need_backend()
    if BACKEND == "cv2":
        img = cv2.imread(path)
        if img is None:
            raise OSError(f"cannot read image: {path}")
        return img
    img = Image.open(path)
    img.load()
    return img.convert("RGB")


def image_size(img) -> tuple:
    """(width, height)"""
    if BACKEND == "cv2":
        return img.shape[1], img.shape[0]
    return img.size


def clamp_box(box, width: int, height: int, pad: float = 0.0):
    """Box [x1,y1,x2,y2] grown by pad*(w,h) on each side and clamped to the frame.
    Returns ints (x1,y1,x2,y2) or None if the result is empty / the box is invalid."""
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - pad * w, x2 + pad * w
    y1, y2 = y1 - pad * h, y2 + pad * h
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def crop_data_url(img, box, pad: float = 0.12, min_side: int = 336, quality: int = 92):
    """Crop a person (with padding, inside the frame), upscale tiny crops so the longer
    side is at least min_side, and return a base64 JPEG data URL. None if the box is invalid."""
    width, height = image_size(img)
    cb = clamp_box(box, width, height, pad)
    if cb is None:
        return None
    x1, y1, x2, y2 = cb
    cw, ch = x2 - x1, y2 - y1
    scale = min_side / max(cw, ch) if min_side and max(cw, ch) < min_side else 1.0
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    if BACKEND == "cv2":
        crop = img[y1:y2, x1:x2]
        if scale != 1.0:
            crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        raw = buf.tobytes()
    else:
        crop = img.crop((x1, y1, x2, y2))
        if scale != 1.0:
            crop = crop.resize((nw, nh), Image.BICUBIC)
        bio = io.BytesIO()
        crop.save(bio, format="JPEG", quality=quality)
        raw = bio.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def annotate_and_save(img, items, header: str, out_path: str, banner_lines=(), banner_kind="ok"):
    """Draw boxes with labels on a COPY of img and save it as JPEG.
    items: list of (box, label, kind) with kind in PALETTE.
    banner_lines: extra text lines under the header (e.g. people list in frame mode)."""
    _need_backend()
    width, height = image_size(img)
    scale = max(0.6, height / 1080.0)
    if BACKEND == "cv2":
        vis = img.copy()
        font, th = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(2 * scale)))
        fs = 0.8 * scale

        def text(x, y, s, rgb):
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.putText(vis, s, (x, y), font, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(vis, s, (x, y), font, fs, bgr, th, cv2.LINE_AA)

        for box, label, kind in items:
            rgb = PALETTE.get(kind, PALETTE["ok"])
            bgr = (rgb[2], rgb[1], rgb[0])
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), bgr, max(2, int(round(3 * scale))))
            text(max(0, x1), max(int(24 * scale), y1 - 8), label, rgb)
        line_h = int(34 * scale)
        text(10, line_h, header, PALETTE[banner_kind])
        for i, s in enumerate(banner_lines, 2):
            text(10, line_h * i, s, (255, 255, 255))
        ok = cv2.imwrite(out_path, vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise OSError(f"cannot write {out_path}")
        return

    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    size = max(14, int(round(22 * scale)))
    try:
        font = ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()

    def text(x, y, s, rgb):
        draw.text((x, y), s, fill=rgb, font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    for box, label, kind in items:
        rgb = PALETTE.get(kind, PALETTE["ok"])
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=max(2, int(round(3 * scale))))
        text(max(0, x1), max(0, y1 - size - 6), label, rgb)
    line_h = int(size * 1.5)
    text(10, 6, header, PALETTE[banner_kind])
    for i, s in enumerate(banner_lines, 1):
        text(10, 6 + line_h * i, s, (255, 255, 255))
    vis.save(out_path, format="JPEG", quality=92)

"""Cloud inking: redraw a pencil drawing as a publication-ready ink drawing with a generalist
image-editing model reached through OpenRouter (https://openrouter.ai).

This is the optional, parallel counterpart of the local pix2pix-turbo workflow in ink.py. The
drawings LEAVE THE COMPUTER: they are sent to OpenRouter and to the provider of the chosen model.
The API key is never stored by this module (it only lives in the arguments of a running batch).

Prompts, image preparation and registration are copied (not imported) from the benchmark that
selected these models, so the app stays standalone.
"""
import base64
import csv
import hashlib
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

# Verify HTTPS against the OS trust store instead of certifi's bundle: antivirus TLS scanning
# re-signs traffic with a root CA that only the OS knows, and certifi then fails.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

try:
    import cv2
except ImportError:  # registration is optional
    cv2 = None

API_URL = "https://openrouter.ai/api/v1/images"
KEYS_URL = "https://openrouter.ai/keys"
INPUT_LONG_SIDE = 1536          # long side of the drawing sent to the API
MAX_VARIANTS = 4
DEFAULT_WORKERS = 3

# ------------------------------------------------------------------------------ models

# est_usd / est_seconds: indicative values measured in the benchmark (median seconds, mean cost per call).
# The real cost of every call is read from the response (usage.cost).
MODELS = {
    "openai/gpt-image-2": {
        "label": "GPT Image 2", "recommended": True, "est_usd": 0.05, "est_seconds": 27,
        "note": "Best in the benchmark (88% of the drawings usable). About 0.05 $ per image.",
    },
}
CUSTOM_MODEL = {"label": "Custom model", "recommended": False, "est_usd": 0.05, "est_seconds": 30,
                "note": "Not tested."}

# Some providers accept only a subset of aspect ratios (HTTP 400 otherwise) or ignore the requested one,
# so the drawing must already have the ratio they return. Applies to custom model ids too.
MODEL_ASPECTS = {
    "recraft/recraft-v4": ["1:1"],
    "krea/krea-2-medium-turbo": ["1:1"],
    "meta/muse-image": ["3:2"],   # always returns 1920x1280 whatever aspect is asked
    "qwen/qwen-image-3": ["1:1", "1:2", "1:4", "2:1", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5",
                          "5:4", "9:16", "16:9"],
}

# Aspect ratios most providers accept; the drawing is padded with white to the closest one.
ASPECTS = {"1:1": 1.0, "4:3": 4 / 3, "3:2": 1.5, "16:9": 16 / 9, "21:9": 21 / 9,
           "3:4": 3 / 4, "2:3": 2 / 3, "9:16": 9 / 16}

# Request parameters that a provider rejects with HTTP 400 (found empirically).
UNSUPPORTED_PARAMS = {
    "black-forest-labs/flux-3-image": ("output_format",),
    "krea/krea-2-medium-turbo": ("resolution", "aspect_ratio", "output_format"),
    "recraft/recraft-v4": ("resolution", "aspect_ratio", "output_format"),
}

# ------------------------------------------------------------------------------ prompts

_COMMON = (
    "Use stippling only for shading: no hatching, no parallel line shading, no grey tones or grey fills. "
    "Do not add any text, letters, labels or numbers. "
    "Where a fracture interrupts an outline, image 1 has short thin lines or dashes showing how the outline "
    "would continue: keep every one of them, in the same place. "
    "The background must be pure white everywhere outside the drawing: no paper texture, no grey tint, no shadows."
)

# Base prompt without reference image (condition R0 of the benchmark): the best one in the data.
BASE_PROMPT = (
    "Image 1 is a pencil drawing of archaeological pottery. Redraw it as a clean, "
    "publication-ready black-and-white archaeological ink drawing: clean black outlines, "
    "with shadows and volume rendered only with stippling (small black dots). "
    "Sections (the cut profile of walls, rims and handles) are a single thin band between two lines, "
    "exactly as in image 1: keep them completely white inside, with no stippling, no shading, no hatching "
    "and no extra parallel lines. "
    "Preserve EXACTLY the profile, section, proportions and the position of every line of image 1; "
    "do not add, remove or move any element. "
    + _COMMON
)

_STYLE_HEAD = (
    "Draw it as a professional archaeological ink drawing (a 'lucido'), made with a technical pen in black ink: "
    "continuous, confident outlines, thicker for the outer contour and thinner for interior and fracture lines. "
)
_STYLE_TAIL = (
    "Express light and shadow only through the spacing of the dots "
    "(closer together in shadow, sparser towards the light), and leave the lit surfaces plain white. "
    "No grey, no gradients, no smudges, no hatching."
)

STYLES = {
    "fine": {
        "label": "Fine",
        "description": "Small, evenly spaced dots: a delicate, dense tone.",
        "fragment": (
            _STYLE_HEAD
            + "Render volume ONLY with stippling: small round dots of uniform size, evenly and openly spaced with "
              "clearly visible white paper between them. "
            + _STYLE_TAIL
        ),
    },
    "coarse": {
        "label": "Coarse",
        "description": "Larger, bolder, widely spaced dots: every dot clearly visible, closer to a hand-stippled lucido.",
        "fragment": (
            _STYLE_HEAD
            + "Render volume ONLY with stippling, using larger, bolder, perfectly round dots of uniform size, "
              "widely and openly spaced, with plenty of white paper between them, so that each single dot is "
              "clearly visible. "
            + _STYLE_TAIL
        ),
    },
}


def build_prompt(style, extra=""):
    """Frozen base prompt + style fragment (+ the user's optional additional instructions)."""
    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'")
    prompt = BASE_PROMPT + " " + STYLES[style]["fragment"]
    extra = (extra or "").strip()
    if extra:
        prompt += " Additional instructions: " + extra
    return prompt


def prompt_hash(prompt):
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def estimate_cost(model, n_drawings, variants=1):
    """Indicative total in USD (the real cost is reported per call by the API)."""
    info = MODELS.get(model, CUSTOM_MODEL)
    return round(info["est_usd"] * n_drawings * max(1, variants), 4)


def public_catalog():
    """What the UI needs to build the controls."""
    return {
        "models": [{"id": k, **{x: v[x] for x in ("label", "recommended", "est_usd", "est_seconds", "note")}}
                   for k, v in MODELS.items()],
        "styles": [{"id": k, "label": v["label"], "description": v["description"]} for k, v in STYLES.items()],
        "max_variants": MAX_VARIANTS,
        "keys_url": KEYS_URL,
    }


# ------------------------------------------------------------------------------ errors

class CloudError(Exception):
    """kind: auth | credits | ssl | rate | network | bad_request | no_image | aspect | other.
    `fatal` errors stop the whole batch (nothing else would succeed)."""

    FATAL = {"auth", "credits", "ssl"}

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind
        self.fatal = kind in self.FATAL


# ------------------------------------------------------------------------------ image helpers

def _aspect_value(label):
    a, b = label.split(":")
    return int(a) / int(b)


def pad_to_aspect(img, allowed=None):
    """Pad with white to the nearest supported aspect ratio.

    Returns (padded image, aspect label, crop box of the original content in relative [0,1]
    coordinates) so the output can be cropped back."""
    w, h = img.size
    r = w / h
    labels = allowed or list(ASPECTS)
    label = min(labels, key=lambda k: abs(np.log(_aspect_value(k) / r)))
    target = _aspect_value(label)
    if target >= r:
        nw, nh = round(h * target), h
    else:
        nw, nh = w, round(w / target)
    canvas = Image.new("L", (nw, nh), 255)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    canvas.paste(img, (x0, y0))
    return canvas, label, (x0 / nw, y0 / nh, (x0 + w) / nw, (y0 + h) / nh)


def fit_long_side(img, long_side):
    s = long_side / max(img.size)
    if s >= 1:
        return img
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)


def to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def crop_back(out_img, rel_box, size):
    W, H = out_img.size
    box = (round(rel_box[0] * W), round(rel_box[1] * H), round(rel_box[2] * W), round(rel_box[3] * H))
    return out_img.crop(box).resize(size, Image.LANCZOS)


# ---- registration of the output onto the pencil (ECC affine on blurred ink-density maps)

_WORK = 768


def _density(img, long_side):
    s = long_side / max(img.size)
    a = np.array(img.convert("L").resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BOX),
                 dtype=np.float32)
    paper = np.percentile(a, 90)
    d = np.clip((paper - a) / max(paper, 1), 0, 1)
    return cv2.GaussianBlur(d, (0, 0), 3)


def align(inp, out, iters=200):
    """Warp `out` (same canvas size as `inp`) onto `inp`. Falls back to the unmodified output if it fails."""
    out = out.convert("L")
    if out.size != inp.size:
        out = out.resize(inp.size, Image.LANCZOS)
    if cv2 is None:
        return out, {"ok": False}
    ref, mov = _density(inp, _WORK), _density(out, _WORK)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
        cc, warp = cv2.findTransformECC(ref, mov, warp, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        return out, {"ok": False}
    s = max(inp.size) / _WORK
    full = warp.copy()
    full[:, 2] = warp[:, 2] * s     # estimated at WORK px: the linear part is scale-free
    aligned = cv2.warpAffine(np.array(out), full, (inp.width, inp.height),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return Image.fromarray(aligned), {"ok": True, "shift_px": float(np.hypot(*warp[:, 2]) * s)}


# ------------------------------------------------------------------------------ API call

def _tokens(usage):
    """(input, output) tokens from a usage dict if the provider reports them, else (None, None)."""
    if not isinstance(usage, dict):
        return None, None

    def first(*names):
        for n in names:
            v = usage.get(n)
            if isinstance(v, (int, float)):
                return int(v)
        return None
    return first("input_tokens", "prompt_tokens"), first("output_tokens", "completion_tokens")


def call_model(api_key, model, prompt, image, aspect, quality="medium"):
    """One image-edit request. Returns (PIL image, cost USD or None, usage dict, provenance dict).

    Raises CloudError: 401/403 -> auth, 402 -> credits (fatal for the batch), other 4xx -> bad_request;
    429/5xx and network errors are retried three times first."""
    payload = {
        "model": model,
        "prompt": prompt,
        "input_references": [{"type": "image_url", "image_url": {"url": to_data_url(image)}}],
        "resolution": "2K",
        "aspect_ratio": aspect,
        "output_format": "png",
    }
    if model.startswith("openai/"):
        payload["quality"] = quality
    for k in UNSUPPORTED_PARAMS.get(model, ()):
        payload.pop(k, None)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "X-Title": "PyPotteryInk"}
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=300)
        except requests.exceptions.SSLError:
            # typically an antivirus that scans HTTPS: retrying cannot help
            raise CloudError("ssl", "The secure connection to OpenRouter was blocked (certificate check failed). "
                                    "This usually comes from an antivirus that scans HTTPS traffic. "
                                    "Run \"pip install truststore\" in the PyPottery environment, restart the app "
                                    "and try again, or pause the antivirus web shield.")
        except requests.RequestException as e:
            last_err = ("network", f"Network error: {e}")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last_err = ("rate" if r.status_code == 429 else "network", f"HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (401, 403):
            raise CloudError("auth", "OpenRouter rejected the API key. Check that it is correct and still active.")
        if r.status_code == 402:
            raise CloudError("credits", "Your OpenRouter account has no credits left. Add credits and start again.")
        if r.status_code != 200:
            raise CloudError("bad_request", f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        items = data.get("data") or []
        if not items or not items[0].get("b64_json"):
            raise CloudError("no_image", "The model returned no image.")
        img = Image.open(io.BytesIO(base64.b64decode(items[0]["b64_json"])))
        img.load()
        usage = data.get("usage") or {}
        prov = {"provider": data.get("provider"), "response_model": data.get("model"), "attempts": attempt + 1}
        return img, usage.get("cost"), usage, prov
    raise CloudError(last_err[0], last_err[1]) if last_err else CloudError("other", "Failed after retries.")


def ink_image(api_key, model, prompt, src, do_align=True):
    """Ink one pencil drawing (PIL, mode L). Returns (output image of the same size as `src`, info dict)."""
    t0 = time.time()
    allowed = MODEL_ASPECTS.get(model)
    padded, aspect, rel_box = pad_to_aspect(src, allowed)
    sent = fit_long_side(padded, INPUT_LONG_SIDE)
    out, cost, usage, prov = call_model(api_key, model, prompt, sent, aspect)
    # the model must hand back the ratio it was sent; otherwise the crop would be misplaced
    if abs((out.width / out.height) / (sent.width / sent.height) - 1) > 0.03:
        raise CloudError("aspect", f"The model returned an unexpected shape ({out.width}x{out.height}).")
    out = crop_back(out.convert("L"), rel_box, src.size)
    aligned = False
    if do_align:
        out, ainfo = align(src, out)
        aligned = bool(ainfo.get("ok"))
    t_in, t_out = _tokens(usage)
    return out, {"seconds": time.time() - t0, "cost": cost, "usage": usage, "tokens_in": t_in,
                 "tokens_out": t_out, "aligned": aligned, "provider": prov.get("provider")}


# ------------------------------------------------------------------------------ batch

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _save(img, path):
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        img.convert("L").save(path, quality=95)
    else:
        img.convert("L").save(path)


def run_batch(api_key, input_folder, output_dir, model, style, variants=1, extra="", do_align=True,
              budget_usd=None, progress=None, cancel=None, workers=DEFAULT_WORKERS):
    """Ink every image of `input_folder`. Each drawing is sent `variants` times.

    `progress(dict)` is called after every event with: done, total, current, cost_usd, estimated_usd,
    tokens_in, tokens_out, has_tokens, elapsed, eta, failed. `cancel` is a threading.Event: once set no
    new call starts (calls in flight finish). With `budget_usd` no new call starts once the real cost so far,
    plus that of the calls in flight, reaches the cap. Returns a summary dict."""
    variants = max(1, min(int(variants), MAX_VARIANTS))
    cancel = cancel or threading.Event()
    in_dir, out_dir = Path(input_folder), Path(output_dir)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    prompt = build_prompt(style, extra)
    p_hash = prompt_hash(prompt)
    est_each = (MODELS.get(model) or CUSTOM_MODEL)["est_usd"]
    jobs = [(f, v) for f in files for v in range(1, variants + 1)]
    total = len(jobs)

    lock = threading.Lock()
    state = {"done": 0, "failed": 0, "cost": 0.0, "cost_known": 0, "tok_in": 0, "tok_out": 0, "has_tokens": False,
             "inflight": 0, "next": 0, "stop": None, "seconds": [], "errors": [], "rows": [], "current": ""}
    t_start = time.time()

    def emit(current=None):
        if not progress:
            return
        with lock:
            if current is not None:
                state["current"] = current
            done = state["done"] + state["failed"]
            elapsed = time.time() - t_start
            mean = (sum(state["seconds"]) / len(state["seconds"])) if state["seconds"] else None
            remaining = total - done
            eta = (mean * remaining / max(1, min(workers, remaining))) if mean and remaining else None
            snap = {"done": done, "total": total, "failed": state["failed"], "current": state["current"],
                    "cost_usd": round(state["cost"], 4),
                    "estimated_usd": round(est_each * total, 4),
                    "tokens_in": state["tok_in"], "tokens_out": state["tok_out"], "has_tokens": state["has_tokens"],
                    "elapsed": round(elapsed, 1), "eta": round(eta, 1) if eta is not None else None}
        progress(snap)

    def next_job():
        with lock:
            if state["stop"] or cancel.is_set():
                if cancel.is_set() and not state["stop"]:
                    state["stop"] = "cancelled"
                return None
            if budget_usd is not None and state["cost"] + state["inflight"] * est_each >= budget_usd:
                state["stop"] = "budget"
                return None
            if state["next"] >= total:
                return None
            job = jobs[state["next"]]
            state["next"] += 1
            state["inflight"] += 1
            return job

    def worker():
        while True:
            job = next_job()
            if job is None:
                return
            f, v = job
            name = f.name if variants == 1 else f"{f.stem}_v{v}{f.suffix}"
            emit(current=name)
            row = {"file": name, "source": f.name, "variant": v, "model": model, "style": style,
                   "prompt_hash": p_hash, "custom_prompt": bool(extra.strip()), "aligned": "",
                   "seconds": "", "cost_usd": "", "tokens_in": "", "tokens_out": "", "provider": "", "error": ""}
            try:
                src = ImageOps.exif_transpose(Image.open(f)).convert("L")
                out, info = ink_image(api_key, model, prompt, src, do_align)
                _save(out, out_dir / name)
                row.update(aligned=info["aligned"], seconds=round(info["seconds"], 1),
                           cost_usd=info["cost"] if info["cost"] is not None else "",
                           tokens_in=info["tokens_in"] if info["tokens_in"] is not None else "",
                           tokens_out=info["tokens_out"] if info["tokens_out"] is not None else "",
                           provider=info["provider"] or "")
                with lock:
                    state["inflight"] -= 1
                    state["done"] += 1
                    state["seconds"].append(info["seconds"])
                    state["cost"] += info["cost"] if info["cost"] is not None else est_each
                    if info["tokens_in"] is not None or info["tokens_out"] is not None:
                        state["has_tokens"] = True
                        state["tok_in"] += info["tokens_in"] or 0
                        state["tok_out"] += info["tokens_out"] or 0
                    state["rows"].append(row)
            except CloudError as e:
                row["error"] = str(e)
                with lock:
                    state["inflight"] -= 1
                    state["failed"] += 1
                    state["errors"].append({"file": name, "kind": e.kind, "message": str(e)})
                    state["rows"].append(row)
                    if e.fatal and not state["stop"]:
                        state["stop"] = e.kind
            except Exception as e:   # unreadable file, disk error...
                row["error"] = f"{type(e).__name__}: {e}"
                with lock:
                    state["inflight"] -= 1
                    state["failed"] += 1
                    state["errors"].append({"file": name, "kind": "other", "message": row["error"]})
                    state["rows"].append(row)
            emit()

    emit()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, total or 1))) as ex:
        for _ in range(max(1, min(workers, total or 1))):
            ex.submit(worker)

    log_path = out_dir / "logs" / f"cloud_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
    cols = ["file", "source", "variant", "model", "style", "prompt_hash", "custom_prompt", "aligned",
            "seconds", "cost_usd", "tokens_in", "tokens_out", "provider", "error"]
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(state["rows"], key=lambda r: (r["source"], r["variant"])))

    ok_rows = sorted((r for r in state["rows"] if not r["error"]), key=lambda r: (r["source"], r["variant"]))
    processed = [r["file"] for r in ok_rows]
    items = [{"file": r["file"], "source": r["source"], "variant": r["variant"], "seconds": r["seconds"],
              "cost_usd": r["cost_usd"]} for r in ok_rows]
    return {
        "successful": state["done"], "failed": state["failed"], "total": total,
        "average_time": (sum(state["seconds"]) / len(state["seconds"])) if state["seconds"] else 0,
        "cost_usd": round(state["cost"], 4), "tokens_in": state["tok_in"], "tokens_out": state["tok_out"],
        "has_tokens": state["has_tokens"], "stopped": state["stop"],
        "errors": state["errors"][:5], "processed_images": processed, "items": items, "output_dir": str(out_dir),
        "log_file": str(log_path), "prompt_hash": p_hash,
    }

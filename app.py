# --- Crash diagnostics: identical block in every PyPottery app (keep in sync) ---
# Native crashes (torch/OpenCV segfaults) leave no Python traceback, and the
# default excepthooks carry no timestamp or thread name. Output goes to stderr,
# which the PyPottery Launcher already captures in logs/apps/<app>.log.
def _install_crash_diagnostics():
    import faulthandler
    import sys
    import threading
    import time
    import traceback

    try:
        faulthandler.enable()
    except Exception:
        pass  # stderr can be None/unusable on windowless builds

    def _report(kind, exc_type, exc_value, exc_tb, thread_name="MainThread"):
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            sys.stderr.write(f"{stamp} CRITICAL [{thread_name}] {kind}\n{text}")
            sys.stderr.flush()
        except Exception:
            pass

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        _report("Unhandled exception in thread", args.exc_type, args.exc_value,
                args.exc_traceback, getattr(args.thread, "name", "?"))

    sys.excepthook = lambda t, v, tb: _report("Unhandled exception", t, v, tb)
    threading.excepthook = _thread_hook


_install_crash_diagnostics()

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from flask import Flask, render_template, request, jsonify, send_file, session, Response
from werkzeug.utils import secure_filename
from pathlib import Path
from hardware_check import HardwareChecker
from typing import Dict, Optional
import os
import shutil
import base64
import requests
import time
import json
from PIL import Image, ImageOps
import numpy as np
import re
import threading
from queue import Queue

import cloud
from ink import process_folder, process_single_image
from models import _MODELS_CACHE_DIR
from preprocessing import DatasetAnalyzer, apply_recommended_adjustments, check_image_quality
from project_manager import ProjectManager, config_label
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'pypotteryink_secret_key_2025'  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Global progress tracking
progress_queues = {}

def _read_version(default: str) -> str:
    """Read the release version from VERSION (bumped automatically by the
    auto-release GitHub Action on every release), falling back to `default`
    for local/dev runs where that file doesn't exist yet."""
    version_file = Path(__file__).resolve().parent / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip() or default
    except OSError:
        return default


version = _read_version("2.1.0")

# Configuration of models with automatic prompts
MODELS = {
    "10k Model": {
        "description": "General-purpose model for pottery drawings",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/model_10k.pkl?download=true",
        "filename": "model_10k.pkl",
        "prompt": "enhance pottery drawing for publication"
    },
    "6h-MCG Model": {
        "description": "High-quality model for Bronze Age drawings",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MCG.pkl?download=true",
        "filename": "6h-MCG.pkl",
        "prompt": "enhance pottery drawing for publication"
    },
    "6h-MC Model": {
        "description": "High-quality model for Protohistoric and Historic drawings",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MC.pkl?download=true",
        "filename": "6h-MC.pkl",
        "prompt": "enhance pottery drawing for publication"
    },
    "4h-PAINT Model": {
        "description": "Tailored model for Historic and painted pottery",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/4h-PAINT.pkl?download=true",
        "filename": "4h-PAINT.pkl",
        "prompt": "enhance pottery drawing for publication"
    },
        "5h-PAPERGRID Model": {
        "description": "Tailored model for handling paper grid tables (DO NOT SUPPORT SHADOWS)",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/5h_PAPERGRID.pkl?download=true",
        "filename": "5h_PAPERGRID.pkl",
        "prompt": "enhance pottery drawing for publication"
    }
}

# Create models folder if it doesn't exist
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def download_model(model_name, session_id=None):
    """Download the selected model if it doesn't already exist. If session_id
    is given and has a live progress queue, real byte progress (we control
    this download directly, so total/downloaded bytes are always known) is
    reported into it the same way inference progress is."""
    model_info = MODELS[model_name]
    model_path = os.path.join(MODELS_DIR, model_info["filename"])

    if not os.path.exists(model_path):
        try:
            print(f"📥 Downloading {model_name}...")
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            last_report_time = 0

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if session_id and session_id in progress_queues:
                            now = time.time()
                            if now - last_report_time >= 0.15 or downloaded == total_size:
                                last_report_time = now
                                pct = (downloaded / total_size * 100) if total_size else 0
                                mb_down = downloaded / (1024 * 1024)
                                mb_total = total_size / (1024 * 1024)
                                progress_queues[session_id].put({
                                    'type': 'style_model_download',
                                    'progress': pct,
                                    'message': f"Downloading {model_name}: {mb_down:.1f} MB / {mb_total:.1f} MB ({pct:.0f}%)"
                                })

            print(f"✅ {model_name} downloaded successfully!")
            return model_path, model_info["prompt"]
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {str(e)}")
            return None, None
    else:
        print(f"✅ {model_name} already exists")
        return model_path, model_info["prompt"]

# ==================== PROJECTS ====================
# Run history per project (which drawings were inked, with which parameters).
# A drawing is identified by the SHA-1 of its bytes, computed by the server when
# it is imported into the project library.
project_manager = ProjectManager(projects_root="projects")
RUNS_FOLDER = 'temp_runs'   # one scratch folder per run, removed when the run ends

def _move(src, dst):
    """Move a file, replacing dst. os.replace when on the same drive, copy + delete otherwise."""
    try:
        os.replace(src, dst)
    except OSError:
        shutil.copy2(src, dst)
        os.remove(src)

def _stage_selection(project_id, names, use_preprocessed, limit=None):
    """Link (or copy) the selected library drawings into a fresh run folder, which is what the
    unchanged run engines read. Each drawing is read in its preprocessed version when asked for
    and available. Returns (run_dir, input_dir, entries[{name, sha1, version}]). Raises ValueError."""
    if not isinstance(names, list) or not names:
        raise ValueError("Select at least one drawing from the project.")
    names = list(dict.fromkeys(str(n) for n in names))
    if limit and len(names) > limit:
        raise ValueError(f"Select at most {limit} drawings.")
    library = project_manager.get_project(project_id)['drawings']
    run_dir = Path(RUNS_FOLDER) / uuid.uuid4().hex
    input_dir = run_dir / 'input'
    input_dir.mkdir(parents=True)
    entries = []
    try:
        for name in names:
            if name not in library:
                raise ValueError(f"{name} is not in this project's library.")
            src, version = project_manager.input_path(project_id, name, use_preprocessed)
            if src is None:
                raise ValueError(f"{name} is missing from the project folder.")
            try:
                os.link(src, input_dir / name)
            except OSError:
                shutil.copy2(src, input_dir / name)
            entries.append({"name": name, "sha1": library[name]['sha1'], "version": version})
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_dir, input_dir, entries

def _place_outputs(project_id, mode, params, scratch, entries, produced, comparisons=False):
    """Move a run's results from its scratch folder to outputs/<mode>/<label>/. The label comes from the
    settings and from the version (original or preprocessed) each drawing was read in, so the same settings
    replace only their own earlier results and different settings never touch each other.
    `produced` maps a drawing name to the files the run made for it. Returns the run items."""
    items = []
    for entry in entries:
        label = config_label(mode, dict(params, input=entry['version']))
        dest = project_manager.outputs_dir(project_id, mode, label)
        files = [f for f in produced.get(entry['name'], []) if (scratch / f).is_file()]
        if files:
            dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            _move(scratch / f, dest / f)
            comparison = scratch / 'comparisons' / f'comparison_{f}'
            if comparisons and comparison.is_file():
                (dest / 'comparisons').mkdir(exist_ok=True)
                _move(comparison, dest / 'comparisons' / comparison.name)
        items.append({"source": entry['name'], "sha1": entry['sha1'], "version": entry['version'],
                      "config": f"{mode}/{label}", "outputs": files, "status": "ok" if files else "failed"})
    logs = scratch / 'logs'
    if logs.is_dir():
        target = project_manager.outputs_dir(project_id, mode) / 'logs'
        target.mkdir(parents=True, exist_ok=True)
        for f in logs.iterdir():
            _move(f, target / f.name)
    return items

def _rescue_scratch(project_id, mode, scratch):
    """Last resort when results could not be sorted into their folders: keep them all, never lose a run."""
    rescue = project_manager.outputs_dir(project_id, mode) / f"recovered_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copytree(scratch, rescue, dirs_exist_ok=True)
    return str(rescue)

def _mirror_items(project_id, mode, items):
    """Second copy of a run in the project's external folder, if one is set: same folder names."""
    external_dir = project_manager.get_project(project_id)['settings'].get('external_dir')
    if not external_dir:
        return None
    report = {"dir": external_dir, "copied": 0, "error": None}
    by_label = {}
    for item in items:
        by_label.setdefault(item['config'].split('/', 1)[1], []).extend(item['outputs'])
    for label, files in by_label.items():
        if not files:
            continue
        result = _mirror_external(external_dir, label, project_manager.outputs_dir(project_id, mode, label), files)
        report["copied"] += result["copied"]
        report["error"] = report["error"] or result["error"]
    return report

def _result_folder(project_id, mode, items):
    """The folder 'Open output folder' opens: the settings folder, or the mode folder when a run filled two."""
    labels = {i['config'] for i in items if i['outputs']}
    if len(labels) == 1:
        return str(project_manager.outputs_dir(project_id, mode, labels.pop().split('/', 1)[1]))
    return str(project_manager.outputs_dir(project_id, mode))

def _number(value):
    """Parameters arrive from the form as strings; keep them as numbers in the history."""
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value

def _require_project(data):
    """(project_id, None) or (None, error response): every tab works inside a project."""
    project_id = (data or {}).get('project_id')
    if not project_id:
        return None, (jsonify({"success": False, "error": "Open a project first."}), 400)
    if project_manager.get_project_path(project_id) is None:
        return None, (jsonify({"success": False, "error": "Unknown project: open or create it again."}), 400)
    project_manager.ensure_folders(project_id)
    return project_id, None

# Parameters that decide a run's results. They name the project output folder (config_label), and
# the same function serves the run and the browser's "done with these settings?" check.
def _local_params(data, session_model_name=None):
    model = data.get('model_name')
    if model == 'custom':
        model = secure_filename(data.get('custom_model_name') or '') or session_model_name or 'custom'
    return {
        "model": model,
        "patch_size": _number(data.get('patch_size', 512)),
        "overlap": _number(data.get('overlap', 64)),
        "contrast_scale": _number(data.get('contrast_scale', 1.0)),
        "upscale": _number(data.get('upscale', 1)),
        "use_fp16": bool(data.get('use_fp16', False)),
    }

def _cloud_params(data):
    model = (data.get('model') or '').strip()
    if model == 'custom':
        model = (data.get('custom_model') or '').strip()
    try:
        variants = max(1, min(int(data.get('variants', 1)), cloud.MAX_VARIANTS))
    except (TypeError, ValueError):
        variants = 1
    try:
        budget = float(data['budget_usd']) if data.get('budget_usd') not in (None, '') else None
    except (TypeError, ValueError):
        budget = None
    reference = data.get('reference', 'none')
    ref_hash = re.sub(r'[^0-9a-f]', '', str(data.get('reference_hash') or '').lower())[:40]
    return {
        "model": model, "style": data.get('style', 'fine'), "variants": variants,
        "extra_prompt": str(data.get('extra_prompt') or '')[:1000],
        "reference": reference, "reference_hash": ref_hash if reference != 'none' else '',
        "align": bool(data.get('align', False)),
        "budget_usd": budget if budget and budget > 0 else None,
    }

def _mirror_external(external_dir, label, output_dir, filenames):
    """Second copy of a run's output files in the user's own folder, `<external>/<label>/`.
    Never raises: the project copy is the real result, so a missing drive only becomes a warning."""
    target = os.path.join(external_dir, label)
    copied, error = 0, None
    try:
        os.makedirs(target, exist_ok=True)
        for name in filenames:
            src = os.path.join(output_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(target, name))   # same settings = same folder: replaces
                copied += 1
    except Exception as e:
        error = str(e)
    return {"dir": target, "copied": copied, "error": error}

# ==================== AUTO-SHUTDOWN WATCHDOG & BEACON ====================
_shutdown_lock = threading.Lock()
_active_tabs: Dict[str, float] = {}  # tab_id -> timestamp
_shutdown_timer: Optional[threading.Timer] = None
_initial_heartbeat_received = False
_start_time = time.time()

# Disabilitabile con variabile d'ambiente per debug/test:
AUTO_SHUTDOWN_ENABLED = os.environ.get("PYPOTTERY_DISABLE_AUTO_SHUTDOWN", "0") != "1"

# Secondi di grazia alla chiusura prima di terminare (consente anche il refresh F5)
AUTO_SHUTDOWN_GRACE_SECONDS = 5.0

# Timeout per schede orfane senza beacon (60s previene falsi allarmi su schede in background/sleep)
TAB_STALE_TIMEOUT_SECONDS = 60.0

# Tempo concesso all'avvio affinché il browser si apra e si connetta
STARTUP_GRACE_SECONDS = 60.0


def _perform_graceful_shutdown():
    """Rilascia la memoria GPU/RAM e termina il processo di sistema."""
    print("[PyPottery] 🛑 Auto-shutdown: Nessuna scheda attiva. Terminazione processo...")
    try:
        import torch, gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # os._exit(0) termina immediatamente il processo Python, liberando porte e memoria
    os._exit(0)


def _cancel_shutdown_timer_locked():
    global _shutdown_timer
    if _shutdown_timer is not None:
        _shutdown_timer.cancel()
        _shutdown_timer = None


def _arm_shutdown_timer_locked(delay_seconds: float = AUTO_SHUTDOWN_GRACE_SECONDS):
    global _shutdown_timer
    if not AUTO_SHUTDOWN_ENABLED:
        return
    if _shutdown_timer is not None:
        _shutdown_timer.cancel()
    _shutdown_timer = threading.Timer(delay_seconds, _perform_graceful_shutdown)
    _shutdown_timer.daemon = True
    _shutdown_timer.start()


def _watchdog_loop():
    """Loop di background per gestire crash o disconnessioni anomale del browser."""
    while True:
        time.sleep(3.0)
        if not AUTO_SHUTDOWN_ENABLED:
            continue

        now = time.time()
        if not _initial_heartbeat_received:
            if now - _start_time < STARTUP_GRACE_SECONDS:
                continue
            with _shutdown_lock:
                if not _initial_heartbeat_received and _shutdown_timer is None:
                    print("[PyPottery] ⚠️ Nessun browser connesso entro la finestra di avvio.")
                    _arm_shutdown_timer_locked(10.0)
            continue

        with _shutdown_lock:
            stale_keys = [k for k, v in _active_tabs.items() if now - v > TAB_STALE_TIMEOUT_SECONDS]
            for k in stale_keys:
                del _active_tabs[k]

            if not _active_tabs and _shutdown_timer is None:
                print(f"[PyPottery] Watchdog: Tutte le schede sono chiuse. Arresto programmato ({AUTO_SHUTDOWN_GRACE_SECONDS}s)...")
                _arm_shutdown_timer_locked(AUTO_SHUTDOWN_GRACE_SECONDS)


threading.Thread(target=_watchdog_loop, daemon=True, name="AutoShutdownWatchdog").start()


@app.route('/api/heartbeat', methods=['POST'])
def handle_heartbeat():
    """Ping periodico dalla scheda attiva del browser."""
    global _initial_heartbeat_received
    data = request.get_json(silent=True) or {}
    tab_id = data.get('tab_id') or request.remote_addr or 'default'
    now = time.time()

    with _shutdown_lock:
        _initial_heartbeat_received = True
        _active_tabs[tab_id] = now
        _cancel_shutdown_timer_locked()

    return jsonify({'status': 'ok', 'active_tabs': len(_active_tabs)})


@app.route('/api/beacon_shutdown', methods=['POST'])
def handle_beacon_shutdown():
    """Inviato via navigator.sendBeacon su pagehide alla chiusura definitiva."""
    try:
        raw = request.get_data()
        data = json.loads(raw.decode('utf-8')) if raw else {}
    except Exception:
        data = request.get_json(silent=True) or {}

    tab_id = data.get('tab_id')
    with _shutdown_lock:
        if tab_id and tab_id in _active_tabs:
            del _active_tabs[tab_id]
        elif not tab_id and _active_tabs:
            if len(_active_tabs) <= 1:
                _active_tabs.clear()

        if not _active_tabs:
            print(f"[PyPottery] Beacon di chiusura ricevuto. Nessuna scheda attiva. Arresto in {AUTO_SHUTDOWN_GRACE_SECONDS}s...")
            _arm_shutdown_timer_locked(AUTO_SHUTDOWN_GRACE_SECONDS)

    return Response(status=204)


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', version=version, models=MODELS, cloud=cloud.public_catalog())

@app.route('/api/system-info')
def get_system_info():
    """Get system hardware information (CPU cores, platform, GPU, MPS)"""
    import platform
    info = {
        'cpu': {
            'cores': os.cpu_count() or 1,
            'platform': platform.system(),
            'arch': platform.machine()
        },
        'gpu': {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_names': []
        },
        'mps': {
            'mps_available': False
        }
    }
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu']['cuda_available'] = True
            info['gpu']['gpu_count'] = torch.cuda.device_count()
            info['gpu']['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['mps']['mps_available'] = True
    except Exception:
        pass
    return jsonify(info)

@app.route('/api/hardware-check', methods=['GET'])
def hardware_check():
    """Check hardware capabilities"""
    try:
        print("Hardware check called")  # Debug log
        checker = HardwareChecker()
        structured_report = checker.get_structured_report()
        
        # Extract hardware availability info
        hardware = {
            "cuda_available": checker.info.get("has_cuda", False),
            "mps_available": checker.info.get("has_mps", False),
            "cpu_available": True  # CPU is always available
        }
        
        print(f"Hardware info: {hardware}")  # Debug log
        
        return jsonify({
            "success": True, 
            "report": structured_report,
            "hardware": hardware
        })
    except Exception as e:
        print(f"Hardware check error: {str(e)}")  # Debug log
        return jsonify({"success": False, "error": str(e)}), 500

def _is_sdturbo_cached():
    """Check if the HF cache (shared across the suite, or local when standalone -
    see models.py) contains the sd-turbo model files."""
    cache_dir = os.path.join(_MODELS_CACHE_DIR, 'hub')
    if not os.path.exists(cache_dir):
        return False
    for item in os.listdir(cache_dir):
        if 'sd-turbo' in item.lower() or 'stabilityai' in item.lower():
            # Check if it has substantial content (not just refs)
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path):
                snapshots_dir = os.path.join(item_path, 'snapshots')
                if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
                    return True
    return False


def _find_sdturbo_blobs_dir():
    """Locate the sd-turbo cache entry's blobs/ dir (the actual file content -
    snapshots/ only holds symlinks to these), or None if not created yet."""
    cache_dir = os.path.join(_MODELS_CACHE_DIR, 'hub')
    if not os.path.exists(cache_dir):
        return None
    for item in os.listdir(cache_dir):
        if 'sd-turbo' in item.lower():
            blobs_dir = os.path.join(cache_dir, item, 'blobs')
            if os.path.isdir(blobs_dir):
                return blobs_dir
    return None


# huggingface_hub only prints download progress to the console (invisible to
# the web UI), and from_pretrained() doesn't expose a byte-progress callback.
# Instead, poll how many bytes have actually landed on disk and compare
# against this known total (unet + text_encoder + vae fp32 safetensors - no
# variant="fp16" is requested anywhere, so the fp32 weights are what's
# downloaded; measured via the HF repo's file listing).
_SDTURBO_ESTIMATED_TOTAL_BYTES = int(4.95 * 1024 ** 3)


def _monitor_sdturbo_download(session_id, stop_event):
    """Background thread: report sd-turbo download progress into the same
    SSE queue used for inference progress, tagged so the frontend can tell
    the two apart. Runs until stop_event is set (model finished loading)."""
    start_time = time.time()
    while not stop_event.is_set():
        blobs_dir = _find_sdturbo_blobs_dir()
        total_bytes = 0
        if blobs_dir:
            for root, _dirs, files in os.walk(blobs_dir):
                for fname in files:
                    try:
                        total_bytes += os.path.getsize(os.path.join(root, fname))
                    except OSError:
                        pass

        pct = min(99, int(total_bytes / _SDTURBO_ESTIMATED_TOTAL_BYTES * 100)) if total_bytes else 0
        gb_down = total_bytes / (1024 ** 3)
        gb_total = _SDTURBO_ESTIMATED_TOTAL_BYTES / (1024 ** 3)
        elapsed = max(1.0, time.time() - start_time)
        speed_mbps = (total_bytes / (1024 * 1024)) / elapsed

        if session_id in progress_queues:
            progress_queues[session_id].put({
                'type': 'sdturbo_download',
                'progress': pct,
                'message': f"Downloading sd-turbo model: {gb_down:.2f} GB / {gb_total:.2f} GB ({pct}%) - {speed_mbps:.1f} MB/s"
            })
        stop_event.wait(0.5)


@app.route('/api/check-diffusion-model', methods=['GET'])
def check_diffusion_model():
    """Check if the sd-turbo diffusion model is already cached locally"""
    try:
        cache_dir = os.path.join(_MODELS_CACHE_DIR, 'hub')
        return jsonify({
            "success": True,
            "cached": _is_sdturbo_cached(),
            "cache_dir": cache_dir
        })
    except Exception as e:
        print(f"Diffusion model check error: {str(e)}")
        return jsonify({"success": False, "error": str(e), "cached": False}), 500

@app.route('/api/download-model', methods=['POST'])
def api_download_model():
    """Download a specific model"""
    data = request.json
    model_name = data.get('model_name')
    
    if model_name not in MODELS:
        return jsonify({"success": False, "error": "Invalid model name"}), 400
    
    model_path, prompt = download_model(model_name)
    
    if model_path:
        return jsonify({
            "success": True,
            "model_path": model_path,
            "prompt": prompt,
            "message": f"Model {model_name} ready"
        })
    else:
        return jsonify({"success": False, "error": "Failed to download model"}), 500

@app.route('/api/check-model', methods=['POST'])
def check_model():
    """Check if model is downloaded, download if not"""
    data = request.json
    model_name = data.get('model_name')
    
    if model_name not in MODELS:
        return jsonify({"success": False, "error": "Invalid model name"}), 400
    
    model_info = MODELS[model_name]
    model_path = os.path.join(MODELS_DIR, model_info["filename"])
    
    # If model doesn't exist, download it
    if not os.path.exists(model_path):
        model_path, prompt = download_model(model_name)
        if not model_path:
            return jsonify({"success": False, "error": "Failed to download model"}), 500
    else:
        prompt = model_info["prompt"]
    
    return jsonify({
        "success": True,
        "model_path": model_path,
        "prompt": prompt,
        "downloaded": True
    })

@app.route('/api/get-models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models_list = []
    for name, info in MODELS.items():
        model_path = os.path.join(MODELS_DIR, info["filename"])
        models_list.append({
            "name": name,
            "description": info["description"],
            "size": info["size"],
            "filename": info["filename"],
            "downloaded": os.path.exists(model_path)
        })
    return jsonify({"success": True, "models": models_list})

@app.route('/api/projects', methods=['GET'])
def list_projects():
    return jsonify({"success": True, "projects": project_manager.list_projects()})

@app.route('/api/projects', methods=['POST'])
def create_project():
    data = request.json or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({"success": False, "error": "Give the project a name."}), 400
    try:
        project = project_manager.create_project(name[:80], (data.get('description') or '').strip()[:500])
        return jsonify({"success": True, "project": project})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    project = project_manager.get_project(project_id)
    if project is None:
        return jsonify({"success": False, "error": "Project not found."}), 404
    return jsonify({
        "success": True, "project": project,
        "path": str(project_manager.get_project_path(project_id).resolve()),
        "outputs": {mode: str(project_manager.outputs_dir(project_id, mode)) for mode in ('local', 'cloud')},
    })

@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    if not project_manager.delete_project(project_id):
        return jsonify({"success": False, "error": "Project not found."}), 404
    return jsonify({"success": True})

@app.route('/api/projects/<project_id>/drawings', methods=['GET'])
def project_drawings(project_id):
    drawings = project_manager.list_drawings(project_id)
    if drawings is None:
        return jsonify({"success": False, "error": "Project not found."}), 404
    return jsonify({"success": True, "drawings": drawings})

@app.route('/api/projects/<project_id>/drawings', methods=['POST'])
def import_drawings(project_id):
    """Add drawings to the project library (multipart field `files`). Duplicates are not copied twice."""
    if project_manager.get_project_path(project_id) is None:
        return jsonify({"success": False, "error": "Project not found."}), 404
    project_manager.ensure_folders(project_id)
    report = {"added": [], "renamed": [], "duplicate": [], "skipped": []}
    for file in request.files.getlist('files'):
        if not file or not file.filename:
            continue
        try:
            result = project_manager.import_drawing(project_id, file.filename, file.stream)
        except Exception as e:
            result = {"status": "skipped", "name": file.filename, "reason": str(e)}
        report[result["status"]].append(result)
    return jsonify({"success": True, "report": report, "drawings": project_manager.list_drawings(project_id)})

@app.route('/api/projects/<project_id>/drawings/<path:name>', methods=['DELETE'])
def remove_drawing(project_id, name):
    if not project_manager.remove_drawing(project_id, name):
        return jsonify({"success": False, "error": "Drawing not found."}), 404
    return jsonify({"success": True})

@app.route('/api/projects/<project_id>/thumb/<version>/<path:name>')
def drawing_thumbnail(project_id, version, name):
    try:
        thumb = project_manager.thumbnail(project_id, name, version)
    except Exception:
        thumb = None   # unreadable image: the browser shows its placeholder
    if thumb is None:
        return jsonify({"error": "Not found"}), 404
    return send_file(os.path.abspath(thumb), mimetype='image/png', max_age=0)

@app.route('/api/projects/<project_id>/file/<area>/<path:relpath>')
def project_file(project_id, area, relpath):
    """Files of a project for the galleries: drawings, inked outputs, comparisons, diagnostics.
    The resolved path must stay inside that project area."""
    base = project_manager.area_dir(project_id, area)
    if base is None:
        return jsonify({"error": "Not found"}), 404
    target = (base / relpath).resolve()
    if base not in target.parents or not target.is_file():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(target), max_age=0)

@app.route('/api/projects/<project_id>/status', methods=['POST'])
def project_status(project_id):
    """For the settings on screen: which output folder each drawing would go to, and whether the project
    already inked it there ('same'), only with other settings ('other') or never ('none')."""
    metadata = project_manager.get_project(project_id)
    if metadata is None:
        return jsonify({"success": False, "error": "Project not found."}), 404
    data = request.json or {}
    mode = data.get('mode')
    if mode == 'local':
        params = _local_params(data)
    elif mode == 'cloud':
        params = _cloud_params(data)
    else:
        return jsonify({"success": False, "error": "Unknown mode."}), 400
    use_preprocessed = bool(data.get('use_preprocessed', True))
    labels = {v: f"{mode}/{config_label(mode, dict(params, input=v))}" for v in ('original', 'preprocessed')}
    index = project_manager.processed_index(project_id)
    status = {}
    for name, info in metadata['drawings'].items():
        version = 'preprocessed' if use_preprocessed and info.get('preprocessed') else 'original'
        records = index.get(info.get('sha1'), {})
        if labels[version] in records:
            entry = {"state": "same", "record": records[labels[version]]}
        elif records:
            entry = {"state": "other", "record": max(records.values(), key=lambda r: r.get('timestamp') or '')}
        else:
            entry = {"state": "none"}
        entry["version"] = version
        status[name] = entry
    return jsonify({"success": True, "labels": labels, "drawings": status})

@app.route('/api/projects/<project_id>/settings', methods=['PATCH'])
def project_settings(project_id):
    """Project settings: today only `external_dir`, the optional second copy of every run's outputs."""
    if project_manager.get_project_path(project_id) is None:
        return jsonify({"success": False, "error": "Project not found."}), 404
    data = request.json or {}
    if 'external_dir' not in data:
        return jsonify({"success": False, "error": "Nothing to update."}), 400
    external_dir = str(data['external_dir'] or '').strip()
    if external_dir:
        try:
            os.makedirs(external_dir, exist_ok=True)
        except OSError as e:
            return jsonify({"success": False, "error": f"Cannot use that folder: {e}"}), 400
        external_dir = os.path.abspath(external_dir)
    settings = project_manager.update_settings(project_id, {'external_dir': external_dir})
    return jsonify({"success": True, "settings": settings})

@app.route('/api/upload-custom-model', methods=['POST'])
def upload_custom_model():
    """Handle custom model upload"""
    try:
        if 'model' not in request.files:
            return jsonify({"success": False, "error": "No model file provided"}), 400
        
        model_file = request.files['model']
        if model_file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not model_file.filename.endswith('.pkl'):
            return jsonify({"success": False, "error": "Only .pkl files are supported"}), 400
        
        # Save to temp location
        filename = secure_filename(model_file.filename)
        temp_model_path = os.path.join(MODELS_DIR, 'temp_' + filename)
        model_file.save(temp_model_path)
        
        # Store in session
        session['custom_model_path'] = temp_model_path
        session['custom_model_name'] = filename
        
        return jsonify({
            "success": True,
            "model_path": temp_model_path,
            "model_name": filename,
            "message": f"Custom model '{filename}' uploaded successfully"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/upload-stats', methods=['POST'])
def upload_stats():
    """Handle upload of a .npy statistics file and return its server path"""
    try:
        if 'stats_file' not in request.files:
            return jsonify({"success": False, "error": "No statistics file provided"}), 400

        stats_file = request.files['stats_file']
        if stats_file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not stats_file.filename.lower().endswith('.npy'):
            return jsonify({"success": False, "error": "Only .npy files are supported"}), 400

        project_id = request.form.get('project_id')
        if project_manager.get_project_path(project_id) is None:
            return jsonify({"success": False, "error": "Open a project first."}), 400
        project_manager.ensure_folders(project_id)
        filename = secure_filename(stats_file.filename)
        stats_file.save(str(project_manager.get_project_path(project_id) / 'stats' / filename))

        return jsonify({
            "success": True,
            "stats_name": filename,
            "message": f"Statistics file '{filename}' added to the project"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

local_stops = {}

@app.route('/api/process-images/stop/<session_id>', methods=['POST'])
def process_images_stop(session_id):
    """Stop a local batch after the drawing being inked; what is done so far is kept."""
    event = local_stops.get(session_id)
    if event is None:
        return jsonify({"success": False, "error": "No running batch with this id."}), 404
    event.set()
    return jsonify({"success": True})

@app.route('/api/process-images', methods=['POST'])
def process_images():
    """Ink the selected drawings of the project with the local model. Results go to
    outputs/local/<settings>/ inside the project and the run is added to its history."""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "Invalid request data"}), 400

        project_id, project_error = _require_project(data)
        if project_error:
            return project_error

        model_name = data.get('model_name')
        use_fp16 = data.get('use_fp16', False)
        contrast_scale = data.get('contrast_scale', 1.0)
        patch_size = data.get('patch_size', 512)
        overlap = data.get('overlap', 64)
        upscale = data.get('upscale', 1)
        use_preprocessed = bool(data.get('use_preprocessed', True))

        # Custom model path can be resolved now (no download involved); the
        # preset-model path/prompt (which may require a download) is instead
        # resolved inside the background thread below, so that download's
        # progress can be reported over the same SSE channel instead of
        # blocking this request.
        custom_model_path = None
        if model_name == 'custom':
            custom_model_path = session.get('custom_model_path')
            if not custom_model_path or not os.path.exists(custom_model_path):
                return jsonify({"success": False, "error": "Custom model not uploaded or not found"}), 400
        elif model_name not in MODELS:
            return jsonify({"success": False, "error": "Unknown model."}), 400

        try:
            run_dir, input_dir, entries = _stage_selection(project_id, data.get('drawings'), use_preprocessed)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        scratch = run_dir / 'out'
        run_params = dict(_local_params(data, session.get('custom_model_name')), use_preprocessed=use_preprocessed)

        # Create a unique session ID for this processing job
        session_id = str(int(time.time() * 1000))
        progress_queues[session_id] = Queue()
        stop_event = threading.Event()
        local_stops[session_id] = stop_event

        # Progress callback function with two progress bars
        def progress_callback(progress, message, patch_progress=None, patch_message=None):
            update = {
                'progress': progress * 100,  # Convert to percentage
                'message': message
            }
            if patch_progress is not None:
                update['patch_progress'] = patch_progress * 100
                update['patch_message'] = patch_message
            progress_queues[session_id].put(update)

        # Process folder in background thread
        def process_in_background():
            try:
                if model_name == 'custom':
                    model_path = custom_model_path
                    prompt = "enhance pottery drawing for publication"
                else:
                    model_info = MODELS[model_name]
                    model_path = os.path.join(MODELS_DIR, model_info["filename"])
                    prompt = model_info["prompt"]

                    # If the model is not present, download it here (in the
                    # background thread) so real progress can stream out.
                    if not os.path.exists(model_path):
                        downloaded_path, downloaded_prompt = download_model(model_name, session_id=session_id)
                        if not downloaded_path:
                            progress_queues[session_id].put({
                                'progress': 0,
                                'message': 'Model not downloaded and download failed',
                                'error': True
                            })
                            return
                        model_path = downloaded_path
                        if downloaded_prompt:
                            prompt = downloaded_prompt

                # sd-turbo (the shared diffusion backbone) downloads lazily
                # the first time a model is loaded - monitor the HF cache
                # dir for real progress while process_folder loads it.
                sdturbo_stop_event = threading.Event()
                if not _is_sdturbo_cached():
                    threading.Thread(
                        target=_monitor_sdturbo_download, args=(session_id, sdturbo_stop_event), daemon=True
                    ).start()

                try:
                    results = process_folder(
                        input_folder=str(input_dir),
                        model_path=model_path,
                        prompt=prompt,
                        output_dir=str(scratch),
                        use_fp16=use_fp16,
                        contrast_scale=contrast_scale,
                        patch_size=patch_size,
                        overlap=overlap,
                        upscale=upscale,
                        progress_callback=progress_callback,
                        export_elements=False,  # Removed SVG export
                        export_svg=False,  # Removed SVG export
                        stop_event=stop_event
                    )
                finally:
                    sdturbo_stop_event.set()

                if not results:  # process_folder returns None when the folder holds no images
                    progress_queues[session_id].put({
                        'progress': 0,
                        'message': 'No images found to process.',
                        'error': True
                    })
                    return

                # A stopped run records only the drawings it started; the others stay "to ink"
                not_started = set(results.get('not_started') or [])
                run_entries = [e for e in entries if e['name'] not in not_started]

                # process_folder names each output like its input
                try:
                    items = _place_outputs(project_id, 'local', run_params, scratch, run_entries,
                                           {e['name']: [e['name']] for e in run_entries}, comparisons=True)
                except Exception as e:
                    rescued = _rescue_scratch(project_id, 'local', scratch)
                    raise RuntimeError(f"the results could not be sorted ({e}); they are all in {rescued}")
                external_result = None
                try:
                    external_result = _mirror_items(project_id, 'local', items)
                    extra = {"average_time": round(results.get('average_time', 0), 1)}
                    if not_started:
                        extra["stopped"] = f"stopped by the user, {len(not_started)} not started"
                    if external_result:
                        extra["external"] = external_result
                    if items:
                        project_manager.add_run(project_id, 'local', run_params, items,
                                                output_dir=_result_folder(project_id, 'local', items), extra=extra)
                except Exception as e:  # the images are already inked: a history problem is not a failed run
                    print(f"Could not record the run in project {project_id}: {e}")

                # Paths relative to the project's outputs/ folder, served by /api/projects/<id>/file/outputs/...
                processed_images, comparison_images = [], []
                for item in items:
                    for f in item['outputs']:
                        processed_images.append(f"{item['config']}/{f}")
                        if (project_manager.outputs_dir(project_id, 'local', item['config'].split('/', 1)[1])
                                / 'comparisons' / f'comparison_{f}').is_file():
                            comparison_images.append(f"{item['config']}/comparisons/comparison_{f}")

                # Send final result
                progress_queues[session_id].put({
                    'progress': 100,
                    'message': 'Processing completed!',
                    'completed': True,
                    'results': {
                        'successful': results.get('successful', 0),
                        'failed': results.get('failed', 0),
                        'not_started': len(not_started),
                        # one line per distinct reason, so a run where everything failed says why
                        'errors': sorted(set((results.get('errors') or {}).values()))[:3],
                        'average_time': results.get('average_time', 0),
                        'processed_images': processed_images,
                        'comparison_images': comparison_images,
                        'output_dir': _result_folder(project_id, 'local', items),
                        'folders': sorted({i['config'] for i in items if i['outputs']}),
                        'external': external_result
                    }
                })
            except Exception as e:
                progress_queues[session_id].put({
                    'progress': 0,
                    'message': f'Error: {str(e)}',
                    'error': True
                })
            finally:
                local_stops.pop(session_id, None)
                shutil.rmtree(run_dir, ignore_errors=True)

        # Start background processing
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()

        return jsonify({
            "success": True,
            "session_id": session_id,
            "images": len(entries),
            "message": "Processing started"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------------------------------------------------------------
# Cloud inking (beta): generalist image models through OpenRouter. Parallel to
# the local pix2pix-turbo workflow above, which it does not touch. The drawings
# are sent to OpenRouter and the model provider; the API key lives only in the
# arguments of the running batch (never in the session cookie, a file or a log).
# ---------------------------------------------------------------------------
cloud_cancels = {}
# Style reference of the cloud workflow (kept in its own scratch folder).
CLOUD_REFERENCE_FOLDER = 'temp_cloud_reference'
CLOUD_REFERENCE_FILES = {"sheet": ["sheet"], "pair": ["pencil", "ink"], "style": ["style"]}

def _load_cloud_reference(mode):
    """The saved reference images for `mode`, in the order the prompt expects. Raises ValueError."""
    if mode == 'none':
        return []
    images = []
    for slot in CLOUD_REFERENCE_FILES[mode]:
        path = os.path.join(CLOUD_REFERENCE_FOLDER, f"{mode}_{slot}.png")
        if not os.path.exists(path):
            raise ValueError("Add the reference drawing first (or switch the reference off).")
        images.append(Image.open(path).convert('L'))
    return images

@app.route('/api/cloud/upload-reference', methods=['POST'])
def cloud_upload_reference():
    """Save the optional style reference: mode 'sheet' (field `sheet`), 'pair' (fields `pencil` and `ink`)
    or 'style' (field `style`: one inked drawing)."""
    try:
        mode = request.form.get('mode', '')
        if mode not in CLOUD_REFERENCE_FILES:
            return jsonify({"success": False, "error": "Unknown reference type."}), 400
        os.makedirs(CLOUD_REFERENCE_FOLDER, exist_ok=True)
        for old in os.listdir(CLOUD_REFERENCE_FOLDER):
            os.remove(os.path.join(CLOUD_REFERENCE_FOLDER, old))
        for slot in CLOUD_REFERENCE_FILES[mode]:
            file = request.files.get(slot)
            if not file or not file.filename:
                return jsonify({"success": False, "error": "Missing reference image."}), 400
            try:
                img = ImageOps.exif_transpose(Image.open(file.stream)).convert('L')
            except Exception:
                return jsonify({"success": False, "error": f"{secure_filename(file.filename)} is not a readable image."}), 400
            img.save(os.path.join(CLOUD_REFERENCE_FOLDER, f"{mode}_{slot}.png"))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def _record_cloud_run(project_id, results, run_params, scratch, entries):
    """Move a finished cloud run into the project, add it to the history and mirror it to the external
    folder, if any. Adds the gallery paths to `results`. Never stores the API key."""
    produced, by_file = {}, {}
    for item in results.get('items', []):
        produced.setdefault(item['source'], []).append(item['file'])
        by_file[item['file']] = item
    items = _place_outputs(project_id, 'cloud', run_params, scratch, entries, produced)
    for item in items:
        outs = [by_file[f] for f in item['outputs'] if f in by_file]
        item["seconds"] = round(sum(float(o['seconds'] or 0) for o in outs), 1)
        item["cost_usd"] = round(sum(float(o['cost_usd'] or 0) for o in outs), 4)

    # gallery: each ink version next to the pencil it was made from, as project file paths
    versions = {i['source']: i['version'] for i in items}
    configs = {i['source']: i['config'] for i in items}
    for item in results.get('items', []):
        item['input_path'] = f"{versions.get(item['source'], 'original')}/{item['source']}"
        item['output_path'] = f"{configs.get(item['source'])}/{item['file']}"
    results['output_dir'] = _result_folder(project_id, 'cloud', items)
    results['folders'] = sorted({i['config'] for i in items if i['outputs']})
    results['log_file'] = ''

    try:
        params = dict(run_params, prompt_hash=results.get('prompt_hash'))
        # keep a copy of the style reference with the project, so the run can be reproduced later
        reference = run_params['reference']
        if reference != 'none':
            label = config_label('cloud', dict(run_params, input='original'))
            ref_dir = Path(project_manager.get_project_path(project_id)) / 'references' / label
            ref_dir.mkdir(parents=True, exist_ok=True)
            for slot in CLOUD_REFERENCE_FILES[reference]:
                src = os.path.join(CLOUD_REFERENCE_FOLDER, f"{reference}_{slot}.png")
                if os.path.exists(src):
                    shutil.copy2(src, ref_dir / f"{reference}_{slot}.png")
            params["reference_dir"] = f"references/{label}"
        external_result = _mirror_items(project_id, 'cloud', items)
        results['external'] = external_result
        extra = {"cost_usd": results.get('cost_usd'), "stopped": results.get('stopped')}
        if external_result:
            extra["external"] = external_result
        project_manager.add_run(project_id, 'cloud', params, items, output_dir=results['output_dir'], extra=extra)
    except Exception as e:  # a history problem must never turn a paid, finished run into an error
        print(f"Could not record the cloud run in project {project_id}: {e}")

@app.route('/api/cloud/catalog', methods=['GET'])
def cloud_catalog():
    return jsonify({"success": True, **cloud.public_catalog()})

@app.route('/api/cloud/start', methods=['POST'])
def cloud_start():
    try:
        data = request.json or {}
        project_id, project_error = _require_project(data)
        if project_error:
            return project_error
        api_key = (data.get('api_key') or '').strip()
        if not api_key:
            return jsonify({"success": False, "error": "Enter your OpenRouter API key first."}), 400
        if not data.get('consent'):
            return jsonify({"success": False, "error": "Confirm that you agree to send the drawings to the provider."}), 400

        model = (data.get('model') or '').strip()
        if model == 'custom':
            model = (data.get('custom_model') or '').strip()
        if not re.fullmatch(r'[\w.\-]+/[\w.\-:]+', model):
            return jsonify({"success": False, "error": "Invalid model id (expected provider/model)."}), 400
        style = data.get('style', 'fine')
        if style not in cloud.STYLES:
            return jsonify({"success": False, "error": "Unknown style level."}), 400
        try:
            variants = max(1, min(int(data.get('variants', 1)), cloud.MAX_VARIANTS))
            budget = float(data['budget_usd']) if data.get('budget_usd') not in (None, '') else None
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "Invalid variants or budget value."}), 400
        if budget is not None and budget <= 0:
            budget = None
        extra = str(data.get('extra_prompt') or '')[:1000]
        reference = data.get('reference', 'none')
        if reference not in cloud.REFERENCE_MODES:
            return jsonify({"success": False, "error": "Unknown reference type."}), 400
        try:
            reference_images = _load_cloud_reference(reference)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        do_align = bool(data.get('align', False))
        use_preprocessed = bool(data.get('use_preprocessed', True))

        try:
            run_dir, input_dir, entries = _stage_selection(project_id, data.get('drawings'), use_preprocessed)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        scratch = run_dir / 'out'
        run_params = dict(_cloud_params(data), use_preprocessed=use_preprocessed)
        n_images = len(entries)

        session_id = 'cloud' + str(int(time.time() * 1000))
        progress_queues[session_id] = Queue()
        cancel = threading.Event()
        cloud_cancels[session_id] = cancel

        def on_progress(snap):
            total = max(snap['total'], 1)
            progress_queues[session_id].put({
                'progress': snap['done'] / total * 100,
                'message': f"{snap['done']} of {snap['total']}",
                'cloud': snap,
            })

        def run():
            try:
                results = cloud.run_batch(
                    api_key=api_key, input_folder=str(input_dir), output_dir=str(scratch), model=model,
                    style=style, variants=variants, extra=extra, do_align=do_align, budget_usd=budget,
                    progress=on_progress, cancel=cancel,
                    reference=reference, reference_images=reference_images)
                try:
                    _record_cloud_run(project_id, results, run_params, scratch, entries)
                except Exception as e:   # paid results: keep them even if sorting them fails
                    print(f"Could not sort the cloud results into project {project_id}: {e}")
                    results['output_dir'] = _rescue_scratch(project_id, 'cloud', scratch)
                progress_queues[session_id].put({
                    'progress': 100, 'message': 'Processing completed!', 'completed': True, 'results': results})
            except Exception as e:
                progress_queues[session_id].put({'progress': 0, 'message': f'Error: {e}', 'error': True})
            finally:
                cloud_cancels.pop(session_id, None)
                shutil.rmtree(run_dir, ignore_errors=True)

        threading.Thread(target=run, daemon=True).start()
        return jsonify({
            "success": True, "session_id": session_id, "images": n_images, "variants": variants,
            "estimated_usd": cloud.estimate_cost(model, n_images, variants),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/cloud/cancel/<session_id>', methods=['POST'])
def cloud_cancel(session_id):
    event = cloud_cancels.get(session_id)
    if event is None:
        return jsonify({"success": False, "error": "No running cloud batch with this id."}), 404
    event.set()
    return jsonify({"success": True})

@app.route('/api/progress/<session_id>')
def get_progress(session_id):
    """Get progress updates for a processing session"""
    def generate():
        if session_id not in progress_queues:
            yield f"data: {json.dumps({'error': 'Invalid session'})}\n\n"
            return
        
        queue = progress_queues[session_id]
        while True:
            try:
                # Get progress update with timeout
                update = queue.get(timeout=30)
                yield f"data: {json.dumps(update)}\n\n"
                
                # If completed or error, cleanup and break
                if update.get('completed') or update.get('error'):
                    del progress_queues[session_id]
                    break
            except:
                # Send keepalive
                yield f"data: {json.dumps({'keepalive': True})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/diagnostics', methods=['POST'])
def run_diagnostics():
    """Patch and contrast diagnostics on up to 5 drawings of the project; output in diagnostics/<time>/."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid request data"}), 400

        project_id, project_error = _require_project(data)
        if project_error:
            return project_error

        model_name = data.get('model_name')
        patch_size = data.get('patch_size', 512)
        overlap = data.get('overlap', 64)
        contrast_values_str = str(data.get('contrast_values', '0.75, 1.0, 1.5, 2.0'))
        try:
            contrast_values = [float(x.strip()) for x in contrast_values_str.split(",") if x.strip()]
        except ValueError:
            return jsonify({"success": False, "error": "Contrast values must be numbers separated by commas."}), 400

        # Custom model path can be resolved now (no download involved); the
        # preset-model path/prompt (which may require a download) is instead
        # resolved inside the background thread below, so that download's
        # progress can be reported over the same SSE channel instead of
        # blocking this request.
        custom_model_path = None
        if model_name == 'custom':
            custom_model_path = session.get('custom_model_path')
            if not custom_model_path or not os.path.exists(custom_model_path):
                return jsonify({"success": False, "error": "Custom model not uploaded or not found"}), 400
        elif model_name not in MODELS:
            return jsonify({"success": False, "error": "Unknown model."}), 400

        try:
            run_dir, input_dir, entries = _stage_selection(
                project_id, data.get('drawings'), bool(data.get('use_preprocessed', True)), limit=5)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400

        from ink import run_diagnostics as ink_run_diagnostics

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        diagnostics_dir = project_manager.area_dir(project_id, 'diagnostics') / stamp
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        # Create session for progress reporting
        session_id = str(int(time.time() * 1000))
        progress_queues[session_id] = Queue()

        def diagnostics_background():
            try:
                if model_name == 'custom':
                    model_path = custom_model_path
                    prompt = "enhance pottery drawing for publication"
                else:
                    model_info = MODELS[model_name]
                    model_path = os.path.join(MODELS_DIR, model_info["filename"])
                    prompt = model_info["prompt"]

                    # If the model is not present, download it here (in the
                    # background thread) so real progress can stream out.
                    if not os.path.exists(model_path):
                        downloaded_path, downloaded_prompt = download_model(model_name, session_id=session_id)
                        if not downloaded_path:
                            progress_queues[session_id].put({
                                'progress': 0,
                                'message': 'Model not downloaded and download failed',
                                'error': True
                            })
                            return
                        model_path = downloaded_path
                        if downloaded_prompt:
                            prompt = downloaded_prompt

                progress_queues[session_id].put({'progress': 5, 'message': 'Starting diagnostics...'})

                def progress_cb(update):
                    try:
                        progress_queues[session_id].put(update)
                    except Exception:
                        pass

                # sd-turbo (the shared diffusion backbone) downloads lazily
                # the first time a model is loaded - monitor the HF cache
                # dir for real progress while the diagnostics run loads it.
                sdturbo_stop_event = threading.Event()
                if not _is_sdturbo_cached():
                    threading.Thread(
                        target=_monitor_sdturbo_download, args=(session_id, sdturbo_stop_event), daemon=True
                    ).start()

                try:
                    success = ink_run_diagnostics(
                        input_folder=str(input_dir),
                        model_path=model_path,
                        prompt=prompt,
                        patch_size=patch_size,
                        overlap=overlap,
                        contrast_values=contrast_values,
                        output_dir=str(diagnostics_dir),
                        progress_callback=progress_cb
                    )
                finally:
                    sdturbo_stop_event.set()

                if success:
                    # paths relative to the project's diagnostics/ folder
                    diagnostic_files = sorted(f"{stamp}/{f}" for f in os.listdir(diagnostics_dir)
                                              if f.lower().endswith(('.png', '.jpg', '.jpeg')))
                    progress_queues[session_id].put({
                        'progress': 100,
                        'message': 'Diagnostics completed!',
                        'completed': True,
                        'results': {'diagnostic_files': diagnostic_files, 'output_dir': str(diagnostics_dir)}
                    })
                else:
                    progress_queues[session_id].put({
                        'progress': 0,
                        'message': 'Diagnostics failed',
                        'error': True
                    })
            except Exception as e:
                progress_queues[session_id].put({
                    'progress': 0,
                    'message': f'Diagnostics error: {str(e)}',
                    'error': True
                })
            finally:
                shutil.rmtree(run_dir, ignore_errors=True)

        thread = threading.Thread(target=diagnostics_background)
        thread.daemon = True
        thread.start()

        return jsonify({"success": True, "session_id": session_id, "message": "Diagnostics started"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def _project_stats_path(project_id, name):
    """A statistics file saved in the project (stats/<name>.npy), or None."""
    if not name:
        return None
    name = secure_filename(str(name))
    if not name.endswith('.npy'):
        name += '.npy'
    path = project_manager.get_project_path(project_id) / 'stats' / name
    return path if path.is_file() else None

@app.route('/api/projects/<project_id>/stats', methods=['GET'])
def project_stats(project_id):
    """Statistics files available to this project: its own (stats/) and the ones shipped with the app."""
    path = project_manager.get_project_path(project_id)
    if path is None:
        return jsonify({"success": False, "error": "Project not found."}), 404
    own = sorted(p.name for p in (path / 'stats').glob('*.npy')) if (path / 'stats').is_dir() else []
    bundled = sorted(p.name for p in Path('.').glob('*_stats.npy'))
    return jsonify({"success": True, "project": own, "bundled": bundled})

def _resolve_stats(project_id, data):
    """The statistics a preprocessing run uses: a project file, a bundled file or an uploaded one."""
    source = data.get('stats_source')
    name = str(data.get('stats_name') or '')
    if source == 'project':
        return _project_stats_path(project_id, name)
    if source == 'bundled' and re.fullmatch(r'[\w.-]+_stats\.npy', name) and os.path.isfile(name):
        return Path(name)
    return None

@app.route('/api/calculate-statistics', methods=['POST'])
def calculate_statistics():
    """Reference statistics from drawings of the project (originals), saved in its stats/ folder."""
    try:
        data = request.get_json(silent=True) or {}
        project_id, project_error = _require_project(data)
        if project_error:
            return project_error
        try:
            run_dir, input_dir, entries = _stage_selection(project_id, data.get('drawings'), False)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        try:
            name = secure_filename(str(data.get('name') or '')) or f"stats_{datetime.now():%Y%m%d_%H%M%S}"
            if not name.endswith('.npy'):
                name += '.npy'
            save_path = project_manager.get_project_path(project_id) / 'stats' / name

            analyzer = DatasetAnalyzer()
            distributions = analyzer.analyze_dataset(str(input_dir))
            analyzer.save_analysis(str(save_path))

            summary = {"images_analyzed": len(entries), "statistics_file": name, "distributions": {}}
            for metric_name, stats in distributions.items():
                summary["distributions"][metric_name] = {
                    "mean": float(stats['mean']),
                    "std": float(stats['std']),
                    "min": float(stats['min']),
                    "max": float(stats['max']),
                    "median": float(stats['percentiles'][2])
                }
            return jsonify({"success": True, "summary": summary})
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/preprocess-images', methods=['POST'])
def preprocess_images():
    """Make the preprocessed version of the selected drawings (always from the original). A drawing whose
    values are already in range gets no preprocessed version, so inking keeps using its original."""
    try:
        data = request.get_json(silent=True) or {}
        project_id, project_error = _require_project(data)
        if project_error:
            return project_error
        names = data.get('drawings')
        if not isinstance(names, list) or not names:
            return jsonify({"success": False, "error": "Select at least one drawing from the project."}), 400
        library = project_manager.get_project(project_id)['drawings']
        unknown = [n for n in names if n not in library]
        if unknown:
            return jsonify({"success": False, "error": f"{unknown[0]} is not in this project's library."}), 400

        stats_path = _resolve_stats(project_id, data)
        if stats_path is None:
            return jsonify({"success": False, "error": "Choose a statistics file first."}), 400
        model_stats = DatasetAnalyzer.load_analysis(str(stats_path)).distributions
        stats_label = stats_path.name

        session_id = str(int(time.time() * 1000))
        progress_queues[session_id] = Queue()
        pre_dir = project_manager.area_dir(project_id, 'preprocessed')

        def preprocess_in_background():
            total = len(names)
            adjusted, unchanged, failed = [], [], []
            try:
                for idx, name in enumerate(names, start=1):
                    try:
                        source = project_manager.drawing_path(project_id, name, 'original')
                        with Image.open(source) as img:
                            image = img.convert('RGB')
                        quality_check = check_image_quality(image, model_stats)
                        if quality_check['recommendations']:
                            result = apply_recommended_adjustments(image, model_stats, verbose=False)
                            # write next to the target and swap: a run reading the old file is not affected
                            tmp = pre_dir / f".tmp-{uuid.uuid4().hex}{Path(name).suffix}"
                            result.save(tmp)
                            os.replace(tmp, pre_dir / name)
                            project_manager.set_preprocessed(project_id, name, {
                                'at': datetime.now().isoformat(timespec='seconds'), 'stats': stats_label,
                                'recommendations': [str(r) for r in quality_check['recommendations']]})
                            adjusted.append(name)
                        else:
                            # already in range: drop an older preprocessed version made with other statistics
                            if project_manager.drawing_path(project_id, name, 'preprocessed') is not None:
                                project_manager.set_preprocessed(project_id, name, None)
                                (pre_dir / name).unlink(missing_ok=True)
                            unchanged.append(name)
                    except Exception as e:
                        failed.append({'name': name, 'error': str(e)})
                    progress_queues[session_id].put({
                        'progress': int(idx / total * 100),
                        'message': f'Checked {idx}/{total} drawings',
                        'processed': idx,
                        'adjusted': len(adjusted)
                    })

                progress_queues[session_id].put({
                    'progress': 100,
                    'message': 'Preprocessing completed',
                    'completed': True,
                    'results': {'processed': total - len(failed), 'adjusted': len(adjusted),
                                'adjusted_names': adjusted, 'unchanged': unchanged, 'failed': failed,
                                'stats': stats_label}
                })
            except Exception as e:
                progress_queues[session_id].put({
                    'progress': 0,
                    'message': f'Error during preprocessing: {str(e)}',
                    'error': True
                })

        thread = threading.Thread(target=preprocess_in_background)
        thread.daemon = True
        thread.start()
        return jsonify({"success": True, "session_id": session_id, "message": "Preprocessing started"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/select-directory', methods=['POST'])
def select_directory():
    """Open native directory picker dialog"""
    try:
        import platform
        import subprocess
        
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            # Use AppleScript for macOS - runs in separate process, avoiding threading issues
            script = '''
            tell application "System Events"
                activate
                set folderPath to choose folder with prompt "Select Output Directory" default location (path to home folder)
                return POSIX path of folderPath
            end tell
            '''
            try:
                result = subprocess.run(
                    ['osascript', '-e', script],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    directory = result.stdout.strip()
                    if directory:
                        return jsonify({
                            "success": True,
                            "directory": directory
                        })
                else:
                    return jsonify({
                        "success": False,
                        "error": "No directory selected"
                    }), 400
            except subprocess.TimeoutExpired:
                return jsonify({
                    "success": False,
                    "error": "Dialog timeout"
                }), 400
        else:
            # Use tkinter for other platforms (Windows, Linux)
            from tkinter import Tk, filedialog
            
            # Create a root window and hide it
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            # Open directory picker
            directory = filedialog.askdirectory(
                title="Select Output Directory",
                mustexist=False
            )
            
            root.destroy()
            
            if directory:
                return jsonify({
                    "success": True,
                    "directory": directory
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "No directory selected"
                }), 400
                
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/open-folder', methods=['POST'])
def open_folder():
    """Open a folder in the system file explorer"""
    try:
        data = request.json
        folder_path = data.get('folder_path', 'temp_diagnostics')
        
        # Make path absolute
        if not os.path.isabs(folder_path):
            folder_path = os.path.abspath(folder_path)
        
        if not os.path.exists(folder_path):
            return jsonify({"success": False, "error": "Folder does not exist"}), 404
        
        import subprocess
        import platform
        
        system = platform.system()
        if system == 'Windows':
            os.startfile(folder_path)
        elif system == 'Darwin':  # macOS
            subprocess.run(['open', folder_path])
        else:  # Linux
            subprocess.run(['xdg-open', folder_path])
        
        return jsonify({"success": True, "message": "Folder opened"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', os.environ.get('PYPOTTERY_PORT', 5003)))

    print("Starting PyPotteryInk Flask Application...")
    print(f"Version: {version}")
    print(f"Models directory: {MODELS_DIR}")

    # Open browser automatically
    import webbrowser
    import threading

    def open_browser():
        # Wait a bit for the server to start
        import time
        time.sleep(1.5)
        webbrowser.open(f'http://127.0.0.1:{port}')

    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Flask app
    app.run(debug=True, host='127.0.0.1', port=port, use_reloader=False)

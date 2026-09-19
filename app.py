import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from flask import Flask, render_template, request, jsonify, send_file, session, Response
from pathlib import Path
from hardware_check import HardwareChecker
from typing import Dict, Optional
import os
import shutil
import base64
import requests
import time
import json
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import threading
from queue import Queue

from ink import process_folder, process_single_image
from models import _MODELS_CACHE_DIR
from preprocessing import DatasetAnalyzer, apply_recommended_adjustments, check_image_quality

app = Flask(__name__)
app.secret_key = 'pypotteryink_secret_key_2025'  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['OUTPUT_FOLDER'] = 'temp_output'

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
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

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

def clear_temp_dirs():
    """Clean temporary directories."""
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

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
    return render_template('index.html', version=version, models=MODELS)

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

@app.route('/api/upload-images', methods=['POST'])
def upload_images():
    """Handle image uploads"""
    try:
        clear_temp_dirs()
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({"success": False, "error": "No files uploaded"}), 400
        
        uploaded_files = []
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append(filename)
        
        return jsonify({
            "success": True,
            "files": uploaded_files,
            "count": len(uploaded_files)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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

        filename = secure_filename(stats_file.filename)
        stats_dir = os.path.join(app.config.get('UPLOAD_FOLDER', 'temp_uploads'), 'stats')
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, filename)
        stats_file.save(stats_path)

        return jsonify({
            "success": True,
            "stats_path": stats_path,
            "message": f"Statistics file '{filename}' uploaded successfully"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/process-images', methods=['POST'])
def process_images():
    """Process uploaded images with the selected model"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "Invalid request data"}), 400
        
        model_name = data.get('model_name')
        output_dir = data.get('output_dir', app.config['OUTPUT_FOLDER'])
        use_fp16 = data.get('use_fp16', False)
        contrast_scale = data.get('contrast_scale', 1.0)
        patch_size = data.get('patch_size', 512)
        overlap = data.get('overlap', 64)
        upscale = data.get('upscale', 1)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Store the output directory in session for image serving
        session['last_output_dir'] = output_dir

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

        # Create a unique session ID for this processing job
        session_id = str(int(time.time() * 1000))
        progress_queues[session_id] = Queue()

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
                sdturbo_monitor = None
                if not _is_sdturbo_cached():
                    sdturbo_monitor = threading.Thread(
                        target=_monitor_sdturbo_download, args=(session_id, sdturbo_stop_event), daemon=True
                    )
                    sdturbo_monitor.start()

                try:
                    results = process_folder(
                        input_folder=app.config['UPLOAD_FOLDER'],
                        model_path=model_path,
                        prompt=prompt,
                        output_dir=output_dir,
                        use_fp16=use_fp16,
                        contrast_scale=contrast_scale,
                        patch_size=patch_size,
                        overlap=overlap,
                        upscale=upscale,
                        progress_callback=progress_callback,
                        export_elements=False,  # Removed SVG export
                        export_svg=False  # Removed SVG export
                    )
                finally:
                    sdturbo_stop_event.set()

                # Get processed images
                processed_images = []
                comparison_images = []

                for file in os.listdir(output_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        processed_images.append(file)

                # Check for comparison images
                comparison_dir = os.path.join(output_dir, 'comparisons')
                if os.path.exists(comparison_dir):
                    for file in os.listdir(comparison_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            comparison_images.append(file)

                # Send final result
                progress_queues[session_id].put({
                    'progress': 100,
                    'message': 'Processing completed!',
                    'completed': True,
                    'results': {
                        'successful': results.get('successful', 0),
                        'failed': results.get('failed', 0),
                        'average_time': results.get('average_time', 0),
                        'processed_images': processed_images,
                        'comparison_images': comparison_images,
                        'output_dir': output_dir,
                        'log_file': results.get('log_file', '')
                    }
                })
            except Exception as e:
                progress_queues[session_id].put({
                    'progress': 0,
                    'message': f'Error: {str(e)}',
                    'error': True
                })

        # Start background processing
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "Processing started"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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
    """Run diagnostics on uploaded images"""
    try:
        # Accept JSON or form-encoded data; be resilient if Content-Type is missing
        data = request.get_json(silent=True)
        if data is None:
            # Fallback to form data (e.g., if client used FormData accidentally)
            if request.form:
                data = request.form.to_dict()
            elif request.data:
                try:
                    data = json.loads(request.data.decode('utf-8'))
                except Exception:
                    data = None

        if not data:
            print('Diagnostics called with empty/invalid payload. Headers:', dict(request.headers))
            return jsonify({"success": False, "error": "Invalid request data"}), 400

        print(f"Diagnostics payload: {data}")
        
        model_name = data.get('model_name')
        patch_size = data.get('patch_size', 512)
        overlap = data.get('overlap', 64)
        contrast_values_str = data.get('contrast_values', '0.75, 1.0, 1.5, 2.0')
        
        # Parse contrast values
        contrast_values = [float(x.strip()) for x in contrast_values_str.split(",") if x.strip()]
        
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

        # Run diagnostics in a background thread and report progress via progress_queues
        from ink import run_diagnostics as ink_run_diagnostics

        diagnostics_dir = 'temp_diagnostics'
        os.makedirs(diagnostics_dir, exist_ok=True)

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
                        print(f"Model '{model_name}' not found at {model_path}. Attempting to download...")
                        downloaded_path, downloaded_prompt = download_model(model_name, session_id=session_id)
                        if not downloaded_path:
                            print(f"Failed to download model: {model_name}")
                            progress_queues[session_id].put({
                                'progress': 0,
                                'message': 'Model not downloaded and download failed',
                                'error': True
                            })
                            return
                        model_path = downloaded_path
                        if downloaded_prompt:
                            prompt = downloaded_prompt

                # Notify start
                progress_queues[session_id].put({
                    'progress': 5,
                    'message': 'Starting diagnostics...',
                })

                # Define a progress callback that pushes updates to the queue
                def progress_cb(update):
                    try:
                        # Ensure keys we send are serializable
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

                # Run the diagnostics (this may take time) and pass the callback
                try:
                    success = ink_run_diagnostics(
                        input_folder=app.config['UPLOAD_FOLDER'],
                        model_path=model_path,
                        prompt=prompt,
                        patch_size=patch_size,
                        overlap=overlap,
                        contrast_values=contrast_values,
                        output_dir=diagnostics_dir,
                        progress_callback=progress_cb
                    )
                finally:
                    sdturbo_stop_event.set()

                if success:
                    diagnostic_files = []
                    for file in os.listdir(diagnostics_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            diagnostic_files.append(file)

                    progress_queues[session_id].put({
                        'progress': 100,
                        'message': 'Diagnostics completed!',
                        'completed': True,
                        'results': {'diagnostic_files': diagnostic_files}
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

        thread = threading.Thread(target=diagnostics_background)
        thread.daemon = True
        thread.start()

        print(f"Diagnostics started. session_id={session_id}")

        return jsonify({"success": True, "session_id": session_id, "message": "Diagnostics started"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/calculate-statistics', methods=['POST'])
def calculate_statistics():
    """Calculate statistics from uploaded images"""
    try:
        save_path = request.json.get('save_path', './custom_stats.npy')
        
        analyzer = DatasetAnalyzer()
        distributions = analyzer.analyze_dataset(app.config['UPLOAD_FOLDER'])
        analyzer.save_analysis(save_path)
        
        # Create summary
        summary = {
            "images_analyzed": len(os.listdir(app.config['UPLOAD_FOLDER'])),
            "statistics_file": save_path,
            "distributions": {}
        }
        
        for metric_name, stats in distributions.items():
            summary["distributions"][metric_name] = {
                "mean": float(stats['mean']),
                "std": float(stats['std']),
                "min": float(stats['min']),
                "max": float(stats['max']),
                "median": float(stats['percentiles'][2])
            }
        
        return jsonify({"success": True, "summary": summary})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/preprocess-images', methods=['POST'])
def preprocess_images():
    """Apply preprocessing adjustments to images"""
    try:
        data = request.get_json(silent=True) or {}
        calculate_stats = data.get('calculate_stats', False)
        stats_file = data.get('stats_file', None)
        output_dir = data.get('output_dir', './preprocessed_images')

        os.makedirs(output_dir, exist_ok=True)

        # Determine which statistics to use
        model_stats = None

        if calculate_stats:
            analyzer = DatasetAnalyzer()
            model_stats = analyzer.analyze_dataset(app.config['UPLOAD_FOLDER'])
        elif stats_file and os.path.exists(stats_file):
            analyzer = DatasetAnalyzer.load_analysis(stats_file)
            model_stats = analyzer.distributions
        else:
            return jsonify({"success": False, "error": "No statistics provided"}), 400

        # Prepare background processing with progress reporting
        session_id = str(int(time.time() * 1000))
        progress_queues[session_id] = Queue()

        def preprocess_in_background():
            try:
                files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
                total = len(files)
                processed_count = 0
                adjusted_count = 0

                if total == 0:
                    progress_queues[session_id].put({
                        'progress': 100,
                        'message': 'No images to preprocess',
                        'completed': True,
                        'results': {'processed': 0, 'adjusted': 0, 'output_dir': output_dir}
                    })
                    return

                for idx, filename in enumerate(files, start=1):
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    try:
                        image = Image.open(image_path).convert('RGB')
                        quality_check = check_image_quality(image, model_stats)

                        if quality_check['recommendations']:
                            adjusted_image = apply_recommended_adjustments(image, model_stats, verbose=False)
                            adjusted_count += 1
                        else:
                            adjusted_image = image

                        output_path = os.path.join(output_dir, filename)
                        adjusted_image.save(output_path)
                        processed_count += 1

                        progress = int(processed_count / total * 100)
                        progress_queues[session_id].put({
                            'progress': progress,
                            'message': f'Processed {processed_count}/{total} images',
                            'processed': processed_count,
                            'adjusted': adjusted_count
                        })
                    except Exception as e:
                        # send a partial error message but continue
                        progress_queues[session_id].put({
                            'progress': int(processed_count / max(1, total) * 100),
                            'message': f'Error processing {filename}: {str(e)}'
                        })

                # Finalize
                progress_queues[session_id].put({
                    'progress': 100,
                    'message': 'Preprocessing completed',
                    'completed': True,
                    'results': {'processed': processed_count, 'adjusted': adjusted_count, 'output_dir': output_dir}
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
        print(f"Preprocessing started. session_id={session_id}, output_dir={output_dir}")

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

@app.route('/api/get-image/<folder>/<filename>')
def get_image(folder, filename):
    """Serve processed images"""
    try:
        # Get the output directory from session or use default
        output_dir = session.get('last_output_dir', app.config['OUTPUT_FOLDER'])
        
        if folder == 'output':
            image_path = os.path.join(output_dir, filename)
        elif folder == 'diagnostics':
            image_path = os.path.join('temp_diagnostics', filename)
        elif folder == 'comparisons':
            # Check in the comparisons subfolder of the output directory
            image_path = os.path.join(output_dir, 'comparisons', filename)
            # If not found, try the default location
            if not os.path.exists(image_path):
                image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'comparisons', filename)
        else:
            return jsonify({"error": "Invalid folder"}), 400
        
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/png')
        else:
            return jsonify({"error": f"Image not found: {image_path}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear-diagnostics', methods=['POST'])
def clear_diagnostics():
    """Clear the diagnostics folder before a new run"""
    try:
        diagnostics_dir = 'temp_diagnostics'
        if os.path.exists(diagnostics_dir):
            shutil.rmtree(diagnostics_dir)
        os.makedirs(diagnostics_dir, exist_ok=True)
        return jsonify({"success": True, "message": "Diagnostics folder cleared"})
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

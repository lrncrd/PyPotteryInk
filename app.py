from flask import Flask, render_template, request, jsonify, send_file, session, Response
from pathlib import Path
from hardware_check import HardwareChecker
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
from preprocessing import DatasetAnalyzer, apply_recommended_adjustments, check_image_quality

app = Flask(__name__)
app.secret_key = 'pypotteryink_secret_key_2025'  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['OUTPUT_FOLDER'] = 'temp_output'

# Global progress tracking
progress_queues = {}

version = "2.0.0"

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

def download_model(model_name):
    """Download the selected model if it doesn't already exist"""
    model_info = MODELS[model_name]
    model_path = os.path.join(MODELS_DIR, model_info["filename"])

    if not os.path.exists(model_path):
        try:
            print(f"📥 Downloading {model_name}...")
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
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

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', version=version, models=MODELS)

@app.route('/api/hardware-check', methods=['GET'])
def hardware_check():
    """Check hardware capabilities"""
    try:
        print("Hardware check called")  # Debug log
        checker = HardwareChecker()
        report = checker.generate_report()
        
        # Extract hardware availability info
        hardware = {
            "cuda_available": checker.info.get("has_cuda", False),
            "mps_available": checker.info.get("has_mps", False),
            "cpu_available": True  # CPU is always available
        }
        
        print(f"Hardware info: {hardware}")  # Debug log
        
        return jsonify({
            "success": True, 
            "report": report,
            "hardware": hardware
        })
    except Exception as e:
        print(f"Hardware check error: {str(e)}")  # Debug log
        return jsonify({"success": False, "error": str(e)}), 500

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
        
        # Get model path and prompt
        if model_name == 'custom':
            # Use custom model from session
            model_path = session.get('custom_model_path')
            if not model_path or not os.path.exists(model_path):
                return jsonify({"success": False, "error": "Custom model not uploaded or not found"}), 400
            prompt = "enhance pottery drawing for publication"
        else:
            model_info = MODELS[model_name]
            model_path = os.path.join(MODELS_DIR, model_info["filename"])
            prompt = model_info["prompt"]
            
            if not os.path.exists(model_path):
                return jsonify({"success": False, "error": "Model not downloaded"}), 400
        
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
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "Invalid request data"}), 400
        
        model_name = data.get('model_name')
        patch_size = data.get('patch_size', 512)
        overlap = data.get('overlap', 64)
        contrast_values_str = data.get('contrast_values', '0.75, 1.0, 1.5, 2.0')
        
        # Parse contrast values
        contrast_values = [float(x.strip()) for x in contrast_values_str.split(",") if x.strip()]
        
        # Get model path and prompt
        if model_name == 'custom':
            # Use custom model from session
            model_path = session.get('custom_model_path')
            if not model_path or not os.path.exists(model_path):
                return jsonify({"success": False, "error": "Custom model not uploaded or not found"}), 400
            prompt = "enhance pottery drawing for publication"
        else:
            model_info = MODELS[model_name]
            model_path = os.path.join(MODELS_DIR, model_info["filename"])
            prompt = model_info["prompt"]
            
            if not os.path.exists(model_path):
                return jsonify({"success": False, "error": "Model not downloaded"}), 400
        
        # Import diagnostics function
        from ink import run_diagnostics
        
        diagnostics_dir = 'temp_diagnostics'
        os.makedirs(diagnostics_dir, exist_ok=True)
        
        success = run_diagnostics(
            input_folder=app.config['UPLOAD_FOLDER'],
            model_path=model_path,
            prompt=prompt,
            patch_size=patch_size,
            overlap=overlap,
            contrast_values=contrast_values,
            output_dir=diagnostics_dir
        )
        
        if success:
            diagnostic_files = []
            for file in os.listdir(diagnostics_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    diagnostic_files.append(file)
            
            return jsonify({
                "success": True,
                "diagnostic_files": diagnostic_files
            })
        else:
            return jsonify({"success": False, "error": "Diagnostics failed"}), 500
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
        data = request.json
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
        
        # Process each image
        processed_count = 0
        adjusted_count = 0
        
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image = Image.open(image_path).convert('RGB')
                
                # Check if adjustments are needed
                quality_check = check_image_quality(image, model_stats)
                
                if quality_check['recommendations']:
                    adjusted_image = apply_recommended_adjustments(image, model_stats, verbose=False)
                    adjusted_count += 1
                else:
                    adjusted_image = image
                
                # Save processed image
                output_path = os.path.join(output_dir, filename)
                adjusted_image.save(output_path)
                processed_count += 1
        
        return jsonify({
            "success": True,
            "processed": processed_count,
            "adjusted": adjusted_count,
            "output_dir": output_dir
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/select-directory', methods=['POST'])
def select_directory():
    """Open native directory picker dialog"""
    try:
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

if __name__ == '__main__':
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
        webbrowser.open('http://127.0.0.1:5003')
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start Flask app
    app.run(debug=True, host='127.0.0.1', port=5003, use_reloader=False)

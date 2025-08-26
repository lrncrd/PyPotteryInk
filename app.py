import gradio as gr
import os
import time
import torch
import requests
from PIL import Image
import shutil
import tempfile
import threading
import queue

# Import PyPotteryInk modules
try:
    from ink import process_single_image, process_folder, run_diagnostics
    from postprocessing import binarize_folder_images, control_stippling
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the PyPotteryInk directory")
    exit(1)

# Custom CSS for professional styling
CUSTOM_CSS = """
/* Main container styling */
.container {
    max-width: 1200px;
    margin: 0 auto;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #8B6F47 0%, #6B4E37 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Card styling */
.gr-box {
    border-radius: 12px !important;
    border: 1px solid #e0e0e0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    background: white !important;
    padding: 1.5rem !important;
    margin-bottom: 1rem !important;
}

/* Button styling */
.gr-button-primary {
    background: linear-gradient(135deg, #8B6F47 0%, #6B4E37 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px rgba(107, 78, 55, 0.3) !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(107, 78, 55, 0.4) !important;
}

/* Input field styling */
.gr-input, .gr-dropdown {
    border-radius: 8px !important;
    border: 2px solid #e0e0e0 !important;
    transition: border-color 0.3s ease !important;
}

.gr-input:focus, .gr-dropdown:focus {
    border-color: #8B6F47 !important;
    outline: none !important;
}

/* Accordion styling */
.gr-accordion {
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
    margin-bottom: 1rem !important;
    overflow: hidden !important;
    background: #fafafa !important;
}

/* Status box styling */
#status-box {
    background: #f8f9fa !important;
    border: 2px solid #e0e0e0 !important;
    border-radius: 8px !important;
    font-family: 'Monaco', 'Consolas', monospace !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
}

/* Image container styling */
.image-container {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    border: 2px solid #e0e0e0 !important;
}

/* Slider styling */
.gr-slider {
    margin: 0.5rem 0 !important;
}

/* Model selector card */
.model-card {
    background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #ddd;
}

/* Success/Error messages */
.success-msg {
    color: #2e7d32;
    font-weight: 500;
}

.error-msg {
    color: #d32f2f;
    font-weight: 500;
}

/* Tab styling */
.gr-tab {
    border-radius: 8px 8px 0 0 !important;
    margin-right: 4px !important;
}

.gr-tab-selected {
    background: #8B6F47 !important;
    color: white !important;
}

/* Progress bar styling */
.progress-bar {
    background: #8B6F47 !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header {
        padding: 1rem;
    }
    
    .gr-box {
        padding: 1rem !important;
    }
}
"""

# Model configurations
MODEL_CONFIGS = {
    "10k Model (General Purpose)": {
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/model_10k.pkl?download=true",
        "filename": "model_10k.pkl",
        "description": "🎯 Best for general pottery drawings from various periods",
        "icon": "🏺"
    },
    "6h-MCG Model (Bronze Age)": {
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MCG.pkl?download=true", 
        "filename": "6h-MCG.pkl",
        "description": "🏛️ Optimized for Bronze Age pottery styles",
        "icon": "⚱️"
    },
    "6h-MC Model (Protohistoric)": {
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MC.pkl?download=true",
        "filename": "6h-MC.pkl", 
        "description": "📜 Specialized for Protohistoric and Historic drawings",
        "icon": "🏺"
    },
    "4h-PAINT Model (Historic)": {
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/4h-PAINT.pkl?download=true",
        "filename": "4h-PAINT.pkl",
        "description": "🎨 Designed for Historic and painted pottery",
        "icon": "🎭"
    }
}

def download_model(model_name, progress_callback=None):
    """Download model if not exists"""
    model_config = MODEL_CONFIGS[model_name]
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_config["filename"])
    
    if os.path.exists(model_path):
        if progress_callback:
            progress_callback(f"✅ Model {model_config['filename']} already exists")
        return model_path
    
    try:
        if progress_callback:
            progress_callback(f"📥 Downloading {model_config['filename']}...")
            progress_callback(f"🌐 From: {model_config['url'][:50]}...")
        
        response = requests.get(model_config["url"], stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size > 0 and progress_callback:
            progress_callback(f"📦 Size: {total_size / 1024 / 1024:.1f} MB")
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and progress_callback:
                        progress = downloaded / total_size * 100
                        mb_downloaded = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        # Create a simple progress bar
                        bar_length = 30
                        filled = int(bar_length * progress / 100)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        progress_callback(f"Downloading: {progress:3.0f}%|{bar}| {mb_downloaded:.1f}/{mb_total:.1f}MB")
        
        if progress_callback:
            progress_callback(f"✅ Model downloaded successfully!")
        return model_path
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Failed to download model: {str(e)}")
        return None

def check_cuda_status():
    """Check GPU/acceleration availability"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return f"""
        <div style="background: #e8f5e9; padding: 12px; border-radius: 8px; border: 2px solid #4caf50;">
            <span style="color: #2e7d32; font-weight: bold;">✅ CUDA GPU Acceleration Active</span><br>
            <span style="color: #388e3c; font-size: 0.9em;">{gpu_name}</span>
        </div>
        """
    elif torch.backends.mps.is_available():
        return """
        <div style="background: #e3f2fd; padding: 12px; border-radius: 8px; border: 2px solid #2196f3;">
            <span style="color: #1565c0; font-weight: bold;">✅ Apple Metal GPU Acceleration Active</span><br>
            <span style="color: #1976d2; font-size: 0.9em;">Using Metal Performance Shaders (MPS)</span>
        </div>
        """
    else:
        return """
        <div style="background: #fff3e0; padding: 12px; border-radius: 8px; border: 2px solid #ff9800;">
            <span style="color: #ef6c00; font-weight: bold;">⚠️ CPU Mode</span><br>
            <span style="color: #f57c00; font-size: 0.9em;">Processing will be slower without GPU acceleration</span>
        </div>
        """

def process_image(
    input_image,
    model_name,
    contrast_scale,
    patch_size,
    overlap,
    use_fp16,
    apply_binarization,
    apply_stippling,
    stippling_operation,
    stippling_intensity,
    progress=gr.Progress()
):
    """Process single image with live status updates"""
    start_time = time.time()
    
    if input_image is None:
        yield None, "⚠️ Please upload an image first"
        return
    
    status_messages = []
    
    def update_status(message):
        # Handle both single messages and full console output
        if '\n' in message:
            # This is full console output, replace everything
            lines = message.split('\n')
            status_messages.clear()
            status_messages.extend(lines)
        else:
            # Single message, append
            status_messages.append(message)
        # Keep only last 30 messages to avoid UI overflow
        while len(status_messages) > 30:
            status_messages.pop(0)
        return "\n".join(status_messages)
    
    try:
        # Download model if needed
        yield None, update_status("🔍 Checking model availability...")
        
        # Create a callback that properly updates and yields
        def download_callback(msg):
            status = update_status(msg)
            # Force UI update by yielding
            return status
            
        model_path = download_model(model_name, download_callback)
        if model_path is None:
            yield None, update_status("❌ Failed to download model")
            return
        
        # Create temporary output directory
        temp_dir = tempfile.mkdtemp(prefix="pypotteryink_")
        yield None, update_status("✅ Workspace initialized")
        
        # Get image info
        img_width, img_height = input_image.size
        total_patches = ((img_width - overlap) // (patch_size - overlap) + 1) * \
                       ((img_height - overlap) // (patch_size - overlap) + 1)
        
        yield None, update_status("\n📊 Image Analysis:")
        yield None, update_status(f"  • Dimensions: {img_width}×{img_height} pixels")
        yield None, update_status(f"  • Total patches: {total_patches}")
        # Determine processing mode
        if torch.cuda.is_available():
            mode = "CUDA GPU (FP16)" if use_fp16 else "CUDA GPU (FP32)"
        elif torch.backends.mps.is_available():
            mode = "Apple Metal GPU (FP32)"
        else:
            mode = "CPU (FP32)"
        yield None, update_status(f"  • Processing mode: {mode}")
        
        # Use the new process function with progress callback
        yield None, update_status("\n🚀 Starting AI Processing...")
        
        # Import the new processing function
        try:
            from process_with_status import process_image_with_queue
            
            # Create a queue for status updates
            status_queue = queue.Queue()
            
            # Process in a thread
            result = {}
            
            def process_thread():
                try:
                    output_path = process_image_with_queue(
                        input_image=input_image,
                        model_path=model_path,
                        output_dir=temp_dir,
                        use_fp16=use_fp16 and torch.cuda.is_available(),
                        contrast_scale=contrast_scale,
                        patch_size=patch_size,
                        overlap=overlap,
                        status_queue=status_queue
                    )
                    result['output_path'] = output_path
                    result['success'] = True
                except Exception as e:
                    result['error'] = str(e)
                    result['success'] = False
                    status_queue.put(f"❌ Error: {str(e)}")
            
            # Start processing thread
            thread = threading.Thread(target=process_thread)
            thread.start()
            
            # Collect and yield updates while processing
            all_messages = []
            
            while thread.is_alive() or not status_queue.empty():
                try:
                    # Get messages from queue (non-blocking with timeout)
                    while True:
                        msg = status_queue.get(timeout=0.05)
                        # Handle progress updates - replace last line if it's a progress bar
                        if 'Processing patches:' in msg and all_messages and 'Processing patches:' in all_messages[-1]:
                            all_messages[-1] = msg
                        else:
                            all_messages.append(msg)
                        
                        # Keep last 50 messages
                        if len(all_messages) > 50:
                            all_messages = all_messages[-50:]
                            
                        # Update display
                        yield None, "\n".join(all_messages)
                        
                except queue.Empty:
                    # No new messages, continue
                    pass
                    
                time.sleep(0.05)  # Small delay
            
            # Get any remaining messages
            while not status_queue.empty():
                try:
                    msg = status_queue.get_nowait()
                    all_messages.append(msg)
                    if len(all_messages) > 50:
                        all_messages = all_messages[-50:]
                except queue.Empty:
                    break
                    
            # Final update
            yield None, "\n".join(all_messages)
            
            thread.join()
            
            if result.get('success'):
                output_path = result['output_path']
            else:
                raise Exception(result.get('error', 'Unknown error'))
            
        except ImportError as e:
            # Fallback to original method with simulated console output
            yield None, update_status(f"⚠️ Using standard processing (import error: {e})")
            yield None, update_status("📦 Loading models...")
            from ink import process_single_image
            
            # Show device info
            if torch.cuda.is_available():
                yield None, update_status(f"🔧 Using device: cuda ({torch.cuda.get_device_name(0)})")
            elif torch.backends.mps.is_available():
                yield None, update_status("🔧 Using device: mps (Metal Performance Shaders)")
            else:
                yield None, update_status("🔧 Using device: cpu")
                
            yield None, update_status("⚙️ Initializing Pix2Pix_Turbo...")
            
            output_path = process_single_image(
                input_image_path_or_pil=input_image,
                model_path=model_path,
                output_dir=temp_dir,
                use_fp16=use_fp16 and torch.cuda.is_available(),
                contrast_scale=contrast_scale,
                patch_size=patch_size,
                overlap=overlap
            )
            yield None, update_status("✅ AI processing completed")
        
        # Apply post-processing if requested
        if apply_binarization:
            yield None, update_status("\n🔲 Applying binarization filter...")
            binarize_folder_images(temp_dir)
            binarized_dir = temp_dir + "_binarized"
            if os.path.exists(binarized_dir):
                for f in os.listdir(binarized_dir):
                    if f.endswith(('.png', '.jpg', '.jpeg')):
                        output_path = os.path.join(binarized_dir, f)
                        break
            yield None, update_status("✅ Binarization completed")
        
        if apply_stippling:
            yield None, update_status(f"\n🔴 Applying stippling effect ({stippling_operation})...")
            working_dir = temp_dir + "_binarized" if apply_binarization else temp_dir
            control_stippling(
                working_dir,
                operation=stippling_operation.lower(),
                intensity=stippling_intensity
            )
            stippled_dir = working_dir + "_dotting_modified"
            if os.path.exists(stippled_dir):
                for f in os.listdir(stippled_dir):
                    if f.endswith(('.png', '.jpg', '.jpeg')):
                        output_path = os.path.join(stippled_dir, f)
                        break
            yield None, update_status("✅ Stippling effect applied")
        
        # Load result
        yield None, update_status("\n📸 Finalizing output...")
        result_image = Image.open(output_path)
        
        # Clean up
        for dir_suffix in ["", "_binarized", "_binarized_dotting_modified", "_dotting_modified"]:
            dir_path = temp_dir + dir_suffix
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        
        elapsed_time = time.time() - start_time
        yield None, update_status(f"\n⏱️  Total processing time: {elapsed_time:.1f} seconds")
        yield None, update_status("\n🎉 Success! Your drawing has been transformed.")
        
        yield result_image, "\n".join(status_messages)
        
    except Exception as e:
        # Clean up on error
        if 'temp_dir' in locals():
            for dir_suffix in ["", "_binarized", "_binarized_dotting_modified", "_dotting_modified"]:
                dir_path = temp_dir + dir_suffix
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
        yield None, update_status(f"\n❌ Error: {str(e)}")

# Create professional interface
def create_interface():
    with gr.Blocks(
        title="PyPotteryInk - Professional Archaeological Drawing Processor",
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="stone",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("Inter")
        ),
        css=CUSTOM_CSS
    ) as interface:
        
        # Header
        with gr.Column(elem_classes="container"):
            gr.HTML("""
            <div class="main-header">
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">🏺 PyPotteryInk</h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                    Transform Archaeological Drawings into Publication-Ready Illustrations
                </p>
            </div>
            """)
            
            # GPU Status
            gr.HTML(check_cuda_status())
            
            # Main content area
            with gr.Row():
                # Left column - Input and settings
                with gr.Column(scale=5):
                    # Input section
                    with gr.Group(elem_classes="gr-box"):
                        gr.Markdown("### 📤 Upload Drawing")
                        input_image = gr.Image(
                            label="Input Drawing",
                            type="pil",
                            height=400,
                            elem_classes="image-container"
                        )
                    
                    # Model selection
                    with gr.Group(elem_classes="model-card"):
                        gr.Markdown("### 🤖 AI Model Selection")
                        model_choice = gr.Dropdown(
                            choices=list(MODEL_CONFIGS.keys()),
                            value="10k Model (General Purpose)",
                            label="Select Model",
                            info="Choose based on your pottery type and period"
                        )
                        model_description = gr.Markdown(
                            value=MODEL_CONFIGS["10k Model (General Purpose)"]["description"],
                            elem_classes="model-desc"
                        )
                    
                    # Processing settings
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                contrast_scale = gr.Slider(
                                    minimum=0.5,
                                    maximum=3.0,
                                    value=1.0,
                                    step=0.1,
                                    label="Contrast Enhancement",
                                    info="Adjust input contrast"
                                )
                                patch_size = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    label="Processing Resolution",
                                    info="Higher = better quality, slower"
                                )
                            with gr.Column():
                                overlap = gr.Slider(
                                    minimum=32,
                                    maximum=128,
                                    value=64,
                                    step=16,
                                    label="Patch Overlap",
                                    info="Reduces seams in output"
                                )
                                use_fp16 = gr.Checkbox(
                                    label="Enable FP16 Mode (CUDA only)",
                                    value=True,
                                    info="Half precision - faster on NVIDIA GPUs only"
                                )
                    
                    # Post-processing options
                    with gr.Accordion("🎨 Output Filters", open=False):
                        apply_binarization = gr.Checkbox(
                            label="Binary Output",
                            value=False,
                            info="Convert to pure black and white"
                        )
                        apply_stippling = gr.Checkbox(
                            label="Stippling Effects",
                            value=False,
                            info="Control dot patterns and shading"
                        )
                        
                        with gr.Row(visible=False) as stippling_options:
                            stippling_operation = gr.Radio(
                                choices=["Dilate", "Erode", "Fade"],
                                value="Fade",
                                label="Stippling Type"
                            )
                            stippling_intensity = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Effect Intensity"
                            )
                        
                        # Show/hide stippling options
                        apply_stippling.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[apply_stippling],
                            outputs=[stippling_options]
                        )
                
                # Right column - Output and status
                with gr.Column(scale=5):
                    # Output section
                    with gr.Group(elem_classes="gr-box"):
                        gr.Markdown("### 🎨 Processed Result")
                        output_image = gr.Image(
                            label="Output",
                            height=400,
                            elem_classes="image-container"
                        )
                    
                    # Status section
                    with gr.Group(elem_classes="gr-box"):
                        gr.Markdown("### 📊 Processing Status")
                        process_status = gr.Textbox(
                            label="",
                            lines=8,
                            max_lines=12,
                            elem_id="status-box",
                            show_label=False,
                            value="Ready to process. Upload an image to begin."
                        )
            
            # Process button
            with gr.Row():
                process_btn = gr.Button(
                    "🚀 Transform Drawing",
                    variant="primary",
                    size="lg",
                    elem_classes="gr-button-primary"
                )
            
            # Footer with links
            gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #666;">
                <p style="margin-bottom: 0.5rem;">
                    <a href="https://lrncrd.github.io/PyPotteryInk/" target="_blank" style="color: #8B6F47; text-decoration: none; margin: 0 1rem;">📚 Documentation</a>
                    <a href="https://github.com/lrncrd/PyPotteryInk" target="_blank" style="color: #8B6F47; text-decoration: none; margin: 0 1rem;">🐙 GitHub</a>
                    <a href="https://huggingface.co/lrncrd/PyPotteryInk" target="_blank" style="color: #8B6F47; text-decoration: none; margin: 0 1rem;">🤗 Models</a>
                </p>
                <p style="font-size: 0.9em; opacity: 0.8;">
                    PyPotteryInk v0.0.3 | Powered by AI Diffusion Models
                </p>
            </div>
            """)
        
        # Update model description when selection changes
        model_choice.change(
            fn=lambda m: MODEL_CONFIGS[m]["description"],
            inputs=[model_choice],
            outputs=[model_description]
        )
        
        # Process button click with live updates
        process_btn.click(
            fn=process_image,
            inputs=[
                input_image,
                model_choice,
                contrast_scale,
                patch_size,
                overlap,
                use_fp16,
                apply_binarization,
                apply_stippling,
                stippling_operation,
                stippling_intensity
            ],
            outputs=[output_image, process_status],
            api_name=False,
            show_progress="full"
        )
    
    return interface

# Main execution
if __name__ == "__main__":
    print("Starting PyPotteryInk Professional Interface...")
    print("Opening in browser at: http://127.0.0.1:7860")
    
    # Create and launch interface
    app = create_interface()
    
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_api=False,
            quiet=False
        )
    except Exception as e:
        print(f"Error launching interface: {e}")
        print("Trying alternative launch...")
        app.launch(
            share=True,
            inbrowser=True,
            show_api=False
        )
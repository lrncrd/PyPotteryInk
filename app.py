import gradio as gr
import os
import shutil
from pathlib import Path
from hardware_check import run_hardware_check, HardwareChecker
import requests
from pathlib import Path
from PIL import Image

from ink import run_diagnostics, process_folder  # Make sure the file is saved
from preprocessing import DatasetAnalyzer, apply_recommended_adjustments, process_folder_metrics, visualize_metrics_change, check_image_quality
import numpy as np

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
        "prompt": "enhance Bronze Age pottery drawing for archaeological publication"
    },
    "6h-MC Model": {
        "description": "High-quality model for Protohistoric and Historic drawings",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MC.pkl?download=true",
        "filename": "6h-MC.pkl",
        "prompt": "enhance protohistoric pottery drawing for publication"
    },
    "4h-PAINT Model": {
        "description": "Tailored model for Historic and painted pottery",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/4h-PAINT.pkl?download=true",
        "filename": "4h-PAINT.pkl",
        "prompt": "enhance painted pottery drawing for archaeological publication"
    }
}

# Create models folder if it doesn't exist
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def download_model(model_name):
    """Download the selected model if it doesn't already exist"""
    model_info = MODELS[model_name]
    model_path = os.path.join(MODELS_DIR, model_info["filename"])

    if not os.path.exists(model_path):
        try:
            print(f"📥 Downloading {model_name}...")
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✅ {model_name} downloaded successfully!")
            return model_path, model_info["prompt"]
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {str(e)}")
            return None, ""
    else:
        print(f"✅ {model_name} already exists")
        return model_path, model_info["prompt"]

def get_model_dropdown():
    """Create the description for the dropdown"""
    choices = []
    for name, info in MODELS.items():
        choices.append(f"{name} ({info['size']}) - {info['description']}")
    return choices

# Temporary directories
TEMP_INPUT = "temp_input"
TEMP_OUTPUT = "temp_output"
TEMP_DIAGNOSTICS = "temp_diagnostics"

os.makedirs(TEMP_INPUT, exist_ok=True)
os.makedirs(TEMP_OUTPUT, exist_ok=True)
os.makedirs(TEMP_DIAGNOSTICS, exist_ok=True)

def clear_temp_dirs():
    """Clean temporary directories at startup."""
    for folder in [TEMP_INPUT, TEMP_OUTPUT, TEMP_DIAGNOSTICS]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

clear_temp_dirs()

def load_model_stats(model_name):
    """Load pre-computed model statistics"""
    stats_files = {
        "10k Model": "models/model_10k_stats.npy",
        "6h-MCG Model": "models/6h_MCG_stats.npy",
        "6h-MC Model": "models/6h_MC_stats.npy",
        "4h-PAINT Model": "models/4h_PAINT_stats.npy"
    }

    model_key = model_name.split(" (")[0]  # Extract model name
    stats_file = stats_files.get(model_key)

    if stats_file and os.path.exists(stats_file):
        try:
            stats = np.load(stats_file, allow_pickle=True).item()
            return stats['distributions']
        except Exception as e:
            print(f"Error loading stats for {model_key}: {e}")
            return None
    return None

def run_calculate_statistics_with_viz(input_images, save_path, generate_viz=True):
    """Calculate statistics from images and optionally generate visualization plots"""
    if not input_images:
        return "❌ Please upload images for statistics calculation", None

    if not save_path:
        save_path = "./custom_stats.npy"

    try:
        # Save images to temp folder
        clear_temp_dirs()
        for img in input_images:
            shutil.copy(img.name, TEMP_INPUT)

        from preprocessing import DatasetAnalyzer
        analyzer = DatasetAnalyzer()
        distributions = analyzer.analyze_dataset(TEMP_INPUT)

        # Save statistics
        analyzer.save_analysis(save_path)

        # Create detailed statistics report
        summary = f"""## ✅ Statistics Calculation Completed!

**Images Analyzed:** {len(input_images)}
**Statistics File:** {save_path}

### 📊 Summary Table:

| Metric | Mean | Std Dev | Min | Max | Median |
|--------|------|---------|-----|-----|--------|"""

        # First add table rows
        for metric_name, stats in distributions.items():
            display_name = metric_name.replace('_', ' ').title()
            summary += f"\n| {display_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['percentiles'][2]:.4f} |"

        summary += "\n\n### 📈 Detailed Analysis:\n"

        # Add detailed statistics for each metric
        for metric_name, stats in distributions.items():
            # Format metric name for display
            display_name = metric_name.replace('_', ' ').title()

            summary += f"\n#### {display_name}:\n"
            summary += f"- **Mean**: {stats['mean']:.4f}\n"
            summary += f"- **Std Dev**: {stats['std']:.4f}\n"
            summary += f"- **Min**: {stats['min']:.4f}\n"
            summary += f"- **Max**: {stats['max']:.4f}\n"
            summary += f"- **Percentiles** (5%, 25%, 50%, 75%, 95%):\n"
            summary += f"  - {', '.join([f'{p:.4f}' for p in stats['percentiles']])}\n"
            summary += f"- **Samples**: {stats['n_samples']}\n"

        summary += f"\n### 💾 File saved as: `{save_path}`\n"
        summary += "\nThe statistics file is ready to use in the preprocessing section below."

        # Generate visualizations if requested
        visualization_paths = []
        if generate_viz:
            try:
                # Create visualization directory
                viz_dir = os.path.join(TEMP_DIAGNOSTICS, "statistics_viz")
                os.makedirs(viz_dir, exist_ok=True)

                # Create individual plots for each metric
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches

                for metric_name, stats in distributions.items():
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Get values
                    values = stats['values']

                    # Create histogram with KDE
                    counts, bins, patches = ax.hist(values, bins=30, density=True,
                                                   alpha=0.7, color='skyblue',
                                                   edgecolor='black', linewidth=1.2)

                    # Add KDE curve if we have enough samples
                    if len(values) > 5:
                        from scipy import stats as scipy_stats
                        kde = scipy_stats.gaussian_kde(values)
                        x_range = np.linspace(min(values), max(values), 100)
                        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

                    # Add statistical markers
                    ax.axvline(stats['mean'], color='green', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.4f}')
                    ax.axvline(stats['percentiles'][2], color='orange', linestyle='--', linewidth=2, label=f'Median: {stats["percentiles"][2]:.4f}')

                    # Add shaded regions for percentiles
                    ax.axvspan(stats['percentiles'][1], stats['percentiles'][3],
                              alpha=0.2, color='gray', label='25th-75th percentile')

                    # Labels and title
                    display_name = metric_name.replace('_', ' ').title()
                    ax.set_title(f'Distribution of {display_name}', fontsize=16, fontweight='bold')
                    ax.set_xlabel(display_name, fontsize=12)
                    ax.set_ylabel('Density', fontsize=12)
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    # Save plot
                    plot_path = os.path.join(viz_dir, f'{metric_name}_distribution.png')
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    visualization_paths.append(plot_path)

                # Create a combined overview plot
                n_metrics = len(distributions)
                fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
                if n_metrics == 1:
                    axes = [axes]

                for idx, (metric_name, stats) in enumerate(distributions.items()):
                    ax = axes[idx]
                    values = stats['values']

                    # Box plot
                    bp = ax.boxplot([values], vert=False, patch_artist=True, widths=0.6)
                    bp['boxes'][0].set_facecolor('lightblue')
                    bp['boxes'][0].set_alpha(0.7)

                    # Add mean marker
                    ax.scatter([stats['mean']], [1], color='red', s=100, zorder=5, label='Mean')

                    # Labels
                    display_name = metric_name.replace('_', ' ').title()
                    ax.set_title(display_name, fontsize=14, fontweight='bold')
                    ax.set_xlabel('Value', fontsize=12)
                    ax.set_yticks([])
                    ax.grid(True, axis='x', alpha=0.3)

                plt.tight_layout()
                overview_path = os.path.join(viz_dir, 'all_metrics_overview.png')
                plt.savefig(overview_path, dpi=150, bbox_inches='tight')
                plt.close()

                visualization_paths.insert(0, overview_path)

            except Exception as e:
                print(f"Warning: Could not generate visualizations: {e}")
                summary += f"\n⚠️ Visualization generation failed: {str(e)}"

        return summary, visualization_paths if visualization_paths else None

    except Exception as e:
        return f"❌ Statistics calculation failed: {str(e)}", None

def run_calculate_statistics(input_images, save_path):
    """Calculate statistics from images and save to .npy file"""
    if not input_images:
        return "❌ Please upload images for statistics calculation"

    if not save_path:
        save_path = "./custom_stats.npy"

    try:
        # Save images to temp folder
        clear_temp_dirs()
        for img in input_images:
            shutil.copy(img.name, TEMP_INPUT)

        from preprocessing import DatasetAnalyzer
        analyzer = DatasetAnalyzer()
        distributions = analyzer.analyze_dataset(TEMP_INPUT)

        # Save statistics
        analyzer.save_analysis(save_path)

        # Create detailed statistics report
        summary = f"""## ✅ Statistics Calculation Completed!

**Images Analyzed:** {len(input_images)}
**Statistics File:** {save_path}

### 📊 Detailed Statistics Report:
"""

        # Add detailed statistics for each metric
        for metric_name, stats in distributions.items():
            # Format metric name for display
            display_name = metric_name.replace('_', ' ').title()

            summary += f"\n#### {display_name}:\n"
            summary += f"- **Mean**: {stats['mean']:.4f}\n"
            summary += f"- **Std Dev**: {stats['std']:.4f}\n"
            summary += f"- **Min**: {stats['min']:.4f}\n"
            summary += f"- **Max**: {stats['max']:.4f}\n"
            summary += f"- **Percentiles** (5%, 25%, 50%, 75%, 95%):\n"
            summary += f"  - {', '.join([f'{p:.4f}' for p in stats['percentiles']])}\n"
            summary += f"- **Samples**: {stats['n_samples']}\n"

        summary += f"\n### 💾 File saved as: `{save_path}`\n"
        summary += "\nThe statistics file is ready to use in the preprocessing section below."

        return summary

    except Exception as e:
        return f"❌ Statistics calculation failed: {str(e)}"

def run_preprocessing_adjustment(input_images, stats_file, output_dir, calculate_stats, use_uploaded_stats):
    """Apply preprocessing adjustments to images"""
    if not input_images:
        return "❌ Please upload images for preprocessing", None

    if not output_dir:
        output_dir = "./preprocessed_images"

    # Prepare directories
    clear_temp_dirs()
    os.makedirs(output_dir, exist_ok=True)

    try:
        from preprocessing import DatasetAnalyzer, apply_recommended_adjustments, check_image_quality

        # Determine which statistics to use
        model_stats = None

        if calculate_stats:
            # Calculate statistics from uploaded images
            print("📊 Calculating statistics from uploaded images...")
            # Save images to temp folder first
            for img in input_images:
                shutil.copy(img.name, TEMP_INPUT)

            analyzer = DatasetAnalyzer()
            model_stats = analyzer.analyze_dataset(TEMP_INPUT)
            print("✅ Statistics calculated successfully")

        elif use_uploaded_stats and stats_file:
            # Load statistics from uploaded .npy file
            print("📁 Loading statistics from uploaded file...")
            analyzer = DatasetAnalyzer.load_analysis(stats_file.name)
            model_stats = analyzer.distributions
            print("✅ Statistics loaded successfully")
        else:
            return "❌ Please upload a statistics file (.npy). You can generate one using the 'Calculate Statistics' section above.", None

        # Process each image
        processed_images = []
        results_summary = []

        for idx, img in enumerate(input_images):
            # Load image
            image = Image.open(img.name).convert('RGB')
            original_name = os.path.basename(img.name)

            # Check if adjustments are needed
            quality_check = check_image_quality(image, model_stats)

            if quality_check['recommendations']:
                # Apply adjustments
                adjusted_image = apply_recommended_adjustments(image, model_stats, verbose=False)
                results_summary.append(f"✅ {original_name}: Adjusted")
            else:
                # No adjustments needed
                adjusted_image = image
                results_summary.append(f"ℹ️ {original_name}: No adjustments needed")

            # Save processed image
            output_path = os.path.join(output_dir, original_name)
            adjusted_image.save(output_path)
            processed_images.append(output_path)

        # Limit to maximum 20 images for gallery display
        display_images = processed_images[:20]
        total_processed = len(processed_images)

        if total_processed > 20:
            summary = f"""## ✅ Preprocessing Completed!

**Total Images Processed:** {total_processed}
**Output Directory:** {output_dir}
**Gallery Display:** Showing first 20 images

### Processing Results:
{chr(10).join(results_summary)}
"""
        else:
            summary = f"""## ✅ Preprocessing Completed!

**Total Images Processed:** {total_processed}
**Output Directory:** {output_dir}

### Processing Results:
{chr(10).join(results_summary)}
"""

        return summary, display_images

    except Exception as e:
        return f"❌ Preprocessing failed: {str(e)}", None

def run_hardware_check():
    """Wrapper function for hardware check"""
    try:
        checker = HardwareChecker()
        return checker.generate_report()
    except Exception as e:
        return f"❌ **Hardware check failed:**\n```\n{str(e)}\n```"

def run_gradio_diagnostics(input_images, model_path, prompt, patch_size, overlap, contrast_values_str):
    if not input_images:
        return "❌ Please upload images for diagnostics", None
    if not model_path or not os.path.exists(model_path):
        return "❌ Invalid model path", None

    # Save uploaded images to a temporary folder
    clear_temp_dirs()
    for img in input_images:
        shutil.copy(img.name, TEMP_INPUT)

    # Process contrasts
    try:
        contrast_values = [float(x.strip()) for x in contrast_values_str.split(",") if x.strip()]
        if not contrast_values:
            contrast_values = [1.0]
    except:
        contrast_values = [1.0]

    # Run diagnostics
    success = run_diagnostics(
        input_folder=TEMP_INPUT,
        model_path=model_path,
        prompt=prompt,
        patch_size=patch_size,
        overlap=overlap,
        contrast_values=contrast_values,
        output_dir=TEMP_DIAGNOSTICS
    )

    if success is False:
        return "❌ Diagnostics failed: no valid images found", None

    # Return results
    diagnostic_images = []
    for file in sorted(os.listdir(TEMP_DIAGNOSTICS)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            diagnostic_images.append(os.path.join(TEMP_DIAGNOSTICS, file))

    # Limit to maximum 20 images for gallery display
    display_images = diagnostic_images[:20]
    total_images = len(diagnostic_images)

    if total_images > 20:
        result_text = f"✅ Diagnostics completed successfully!\n📊 Generated {total_images} visualization(s) (showing first 20)"
    else:
        result_text = f"✅ Diagnostics completed successfully!\n📊 Generated {total_images} visualization(s)"

    return result_text, display_images

def open_folder(path):
    """Open a folder in the system's file explorer"""
    import platform
    import subprocess

    if platform.system() == 'Darwin':  # macOS
        subprocess.Popen(['open', path])
    elif platform.system() == 'Windows':  # Windows
        subprocess.Popen(['explorer', path])
    else:  # Linux and others
        subprocess.Popen(['xdg-open', path])

def run_gradio_processing(input_images, model_path, prompt, output_dir, use_fp16, contrast_scale,
                          patch_size, overlap, upscale, export_elements, export_svg, progress=gr.Progress()):
    if not input_images:
        return "❌ No images to process", None, gr.update(visible=False), gr.update(visible=False), ""
    if not model_path or not os.path.exists(model_path):
        return "❌ Invalid model path", None, gr.update(visible=False), gr.update(visible=False), ""

    # Show attribution reminder popup
    popup_message = gr.update(
        value="⚠️ **Please check the 'About & Disclaimer' tab for proper attribution requirements when publishing results with PyPotteryInk.**",
        visible=True
    )

    # Use project directory if available, otherwise use provided path
    if not output_dir:
        output_dir = "./enhanced_pottery"

    # Clean and prepare folders
    clear_temp_dirs()
    os.makedirs(output_dir, exist_ok=True)

    # Copy uploaded images
    for img in input_images:
        shutil.copy(img.name, TEMP_INPUT)

    # Run batch processing
    try:
        # Create a progress callback
        def update_progress(prog_value, status_text):
            progress(prog_value, desc=status_text)

        results = process_folder(
            input_folder=TEMP_INPUT,
            model_path=model_path,
            prompt=prompt,
            output_dir=output_dir,
            use_fp16=use_fp16,
            contrast_scale=contrast_scale,
            patch_size=patch_size,
            overlap=overlap,
            upscale=upscale,
            progress_callback=update_progress,
            export_elements=export_elements,
            export_svg=export_svg
        )

        # Prepare comparison images for gallery
        comparison_images = []
        if "comparison_dir" in results and results["comparison_dir"] and os.path.exists(results["comparison_dir"]):
            for file in sorted(os.listdir(results["comparison_dir"])):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    comparison_images.append(os.path.join(results["comparison_dir"], file))

        # Limit to maximum 20 images for gallery display
        display_images = comparison_images[:20]
        total_comparisons = len(comparison_images)

        if total_comparisons > 20:
            gallery_note = f"• 🖼️ Gallery: Showing first 20 of {total_comparisons} comparison images"
        else:
            gallery_note = f"• 🖼️ Gallery: {total_comparisons} comparison images"

        summary = (
            f"🎉 **Processing completed successfully!**\n\n"
            f"📈 **Results Summary:**\n"
            f"• ✅ Successful: **{results['successful']}** images\n"
            f"• ❌ Failed: **{results['failed']}** images\n"
            f"• ⏱️ Average processing time: **{results['average_time']:.2f}s** per image\n"
            f"• 📁 Output directory: `{output_dir}`\n"
            f"• 📝 Log file: `{results.get('log_file', 'N/A')}`\n"
            f"{gallery_note}"
        )

        # Show open folder button
        open_folder_button = gr.update(visible=True)

        return summary, display_images, popup_message, open_folder_button, output_dir
    except Exception as e:
        return f"❌ **Processing Error:**\n```\n{str(e)}\n```", None, popup_message, gr.update(visible=False), ""

def run_hardware_check():
    """Wrapper function for hardware check"""
    try:
        checker = HardwareChecker()
        return checker.generate_report()
    except Exception as e:
        return f"❌ **Hardware check failed:**\n```\n{str(e)}\n```"

# Function to handle model selection
def on_model_select(selection):
    if not selection:
        return "", ""

    # Extract model name from selection
    model_name = selection.split(" (")[0]
    if model_name in MODELS:
        path, prompt = download_model(model_name)
        return path if path else "", prompt
    return "", ""

with gr.Blocks(
    title="PyPotteryInk",
    css="""
    #attribution-popup {
        background: #fef3c7 !important;
        border: 2px solid #f59e0b !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        animation: fadeIn 0.5s ease-in-out !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    #attribution-popup p {
        color: #92400e !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    """
) as demo:

    # Convert logo image to base64 to embed it directly in HTML
    image_path = os.path.join(os.path.dirname(__file__), "imgs", "pypotteryink.png")
    with open(image_path, "rb") as img_file:
        import base64
        img_data = base64.b64encode(img_file.read()).decode()

    gr.HTML(f"""
    <div style="display: flex; align-items: center; padding: 20px; background: #f9fafb; border-radius: 8px; margin-bottom: 25px; border: 1px solid #e5e7eb;">
        <div style="margin-right: 20px;">
            <img src="data:image/png;base64,{img_data}"
                alt="PyPotteryInk Logo"
                style="border-radius: 8px; width: 64px; height: 64px; object-fit: contain;"/>
        </div>
        <div>
            <h1 style="color: #1f2937; font-size: 2.2em; margin: 0; font-weight: 600;">
                PyPotteryInk
            </h1>
            <p style="color: #6b7280; font-size: 1.1em; margin: 8px 0 0 0;">
                v1.0 - AI-Powered Archaeological Pottery Enhancement
            </p>
        </div>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: Hardware Check
        with gr.Tab("Hardware Check", elem_id="hw-tab"):
            gr.HTML("""
            <div style="background: #f3f4f6; border-left: 4px solid #9ca3af; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #374151; margin-top: 0;">System Requirements Check</h3>
                <p style="color: #4b5563; margin-bottom: 0; font-size: 16px;">
                    This tool requires significant computational resources. Please verify your hardware meets the requirements.
                </p>
            </div>
            """)

            with gr.Row():
                hw_btn = gr.Button("Analyze Hardware", variant="primary", scale=1, size="lg")

            hw_report = gr.Markdown(label="Hardware Analysis Report")
            hw_btn.click(fn=run_hardware_check, outputs=hw_report)

            gr.HTML("""
            <div style="background: #f3f4f6; border-left: 4px solid #9ca3af; padding: 20px; border-radius: 8px; margin-top: 20px;">
                <h4 style="color: #374151; margin-top: 0;">Recommended Specifications:</h4>
                <ul style="color: #4b5563; font-size: 15px;">
                    <li><strong>GPU:</strong> NVIDIA with at least 8GB VRAM (minimum 4GB)</li>
                    <li><strong>CPU:</strong> 4+ modern cores (Intel i5/AMD Ryzen 5 or better)</li>
                    <li><strong>RAM:</strong> 16GB (minimum 8GB)</li>
                    <li><strong>Storage:</strong> Fast SSD (NVMe recommended)</li>
                </ul>
            </div>
            """)

                # TAB 2: Model Diagnostics
        with gr.Tab("Model Diagnostics", elem_id="diag-tab"):
            gr.HTML("""
            <div style="background: #f3f4f6; border-left: 4px solid #9ca3af; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #374151; margin-top: 0;">Pre-Processing Analysis</h3>
                <p style="color: #4b5563; margin-bottom: 0; font-size: 16px;">
                    Run preliminary tests to visualize patch processing and contrast comparisons before full processing.
                </p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    diag_input = gr.File(
                        file_count="multiple",
                        label="Upload Images for Diagnostics",
                        type="filepath",
                        file_types=["image"]
                    )

                with gr.Column(scale=1):
                    model_dropdown_diag = gr.Dropdown(
                        label="Select AI Model",
                        choices=get_model_dropdown(),
                        info="Each model is specialized for different pottery types",
                        value=""
                    )

            # Hidden components to handle the model
            model_path_hidden_diag = gr.Textbox(visible=False)
            model_prompt_hidden_diag = gr.Textbox(visible=False)

            with gr.Row():
                with gr.Column():
                    diag_patch_size = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64,
                        label="Patch Size",
                        info="Size of processing patches"
                    )
                with gr.Column():
                    diag_overlap = gr.Slider(
                        minimum=0, maximum=128, value=64, step=8,
                        label="Patch Overlap",
                        info="Overlap between patches"
                    )

            diag_contrast = gr.Textbox(
                label="Contrast Test Values",
                value="0.75, 1.0, 1.5, 2.0",
                info="Comma-separated values for contrast comparison"
            )

            with gr.Row():
                diag_button = gr.Button("Run Diagnostics", variant="primary", size="lg")

            diag_output_text = gr.Markdown(label="Diagnostic Results")
            diag_output_images = gr.Gallery(
                label="Diagnostic Visualizations",
                show_label=True,
                elem_id="diag-gallery",
                columns=3,
                height="auto",
                object_fit="contain"
            )

            # Event handlers for diagnostics
            model_dropdown_diag.change(
                fn=on_model_select,
                inputs=model_dropdown_diag,
                outputs=[model_path_hidden_diag, model_prompt_hidden_diag]
            )

            diag_button.click(
                fn=run_gradio_diagnostics,
                inputs=[diag_input, model_path_hidden_diag, model_prompt_hidden_diag,
                       diag_patch_size, diag_overlap, diag_contrast],
                outputs=[diag_output_text, diag_output_images]
            )

        # TAB 3: Preprocessing
        with gr.Tab("Preprocessing", elem_id="preprocessing-tab"):
            gr.HTML("""
            <div>
                <h3 style="color: #374151; margin-top: 0;">Image Preprocessing</h3>
                <p style="color: #4b5563; margin-bottom: 0; font-size: 16px;">
                    Calculate statistics from training images and apply preprocessing to optimize images for AI processing.
                </p>
            </div>
            """)

            # SECTION 1: Calculate Statistics (Optional)
            gr.HTML("""
            <div style="background: #f3f4f6; border-left: 4px solid #9ca3af; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #374151; margin-top: 0;">📊 Calculate Statistics (Optional)</h3>
                <p style="color: #4b5563; margin-bottom: 0; font-size: 16px;">
                    Generate statistics from a dataset to create custom preprocessing parameters.
                </p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    stats_calc_input = gr.File(
                        file_count="multiple",
                        label="Upload Training Images for Statistics",
                        type="filepath",
                        file_types=["image"]
                    )

                with gr.Column(scale=1):
                    stats_save_path = gr.Textbox(
                        label="Statistics File Path",
                        value="./custom_stats.npy",
                        info="Where to save calculated statistics"
                    )
                    stats_visualize = gr.Checkbox(
                        label="Generate visualization plots",
                        value=True,
                        info="Create graphs showing the statistical distributions"
                    )

            with gr.Row():
                stats_calc_button = gr.Button("Calculate Statistics", variant="secondary", size="lg")

            stats_calc_output = gr.Markdown(label="Statistics Calculation Results")
            stats_visualization = gr.Gallery(
                label="Statistical Distributions",
                show_label=True,
                columns=2,
                height="auto",
                object_fit="contain"
            )

            stats_calc_button.click(
                fn=lambda imgs, path, viz: run_calculate_statistics_with_viz(imgs, path, viz),
                inputs=[stats_calc_input, stats_save_path, stats_visualize],
                outputs=[stats_calc_output, stats_visualization]
            )

            gr.HTML("<hr style='margin: 30px 0; border: 1px solid #e5e7eb;'>")

            # SECTION 2: Apply Preprocessing (Main)
            gr.HTML("""
            <div style="background: #f3f4f6; border-left: 4px solid #9ca3af; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #374151; margin-top: 0;">🔧 Apply Preprocessing</h3>
                <p style="color: #4b5563; margin-bottom: 0; font-size: 16px;">
                    Process images using calculated or pre-existing statistics to optimize them for AI processing.
                </p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    prep_input = gr.File(
                        file_count="multiple",
                        label="Upload Images to Preprocess",
                        type="filepath",
                        file_types=["image"]
                    )

                with gr.Column(scale=1):
                    prep_stats_file = gr.File(
                        label="Upload Statistics File (.npy)",
                        file_types=[".npy"],
                        #info="Use statistics from section above or upload existing .npy file"
                    )

            with gr.Row():
                with gr.Column():
                    prep_output_dir = gr.Textbox(
                        label="Output Directory",
                        value="./preprocessed_images"
                        #info="Where to save preprocessed images"
                    )

            with gr.Row():
                prep_button = gr.Button("Apply Preprocessing", variant="primary", size="lg")

            prep_output = gr.Markdown(label="Preprocessing Results")
            prep_gallery = gr.Gallery(
                label="Processed Images",
                show_label=True,
                elem_id="prep-gallery",
                columns=3,
                height="auto",
                object_fit="contain"
            )

            # Event handlers
            prep_button.click(
                fn=lambda images, stats_file, output_dir: run_preprocessing_adjustment(
                    images,
                    stats_file,
                    output_dir,
                    False,  # Don't calculate stats in this section
                    True    # Always use uploaded file
                ),
                inputs=[prep_input, prep_stats_file, prep_output_dir],
                outputs=[prep_output, prep_gallery]
            )

        # TAB 4: Batch Processing
        with gr.Tab("Batch Processing", elem_id="proc-tab"):
            gr.HTML("""
            <div style="background: #f3f4f6; border-left: 4px solid #9ca3af; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #374151; margin-top: 0;">Batch Image Enhancement</h3>
                <p style="color: #4b5563; margin-bottom: 0; font-size: 16px;">
                    Process multiple pottery drawings simultaneously with AI enhancement.
                </p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    proc_input = gr.File(
                        file_count="multiple",
                        label="Upload Images to Process",
                        type="filepath",
                        file_types=["image"]
                    )

                with gr.Column(scale=1):
                    proc_model_dropdown = gr.Dropdown(
                        label="Select AI Model",
                        choices=get_model_dropdown(),
                        info="Choose the model best suited for your pottery type",
                        value=""
                    )

            # Hidden components
            proc_model_path_hidden = gr.Textbox(visible=False)
            proc_model_prompt_hidden = gr.Textbox(visible=False)

            with gr.Row():
                with gr.Column():
                    proc_output_dir = gr.Textbox(
                        label="Output Directory",
                        value="./enhanced_pottery",
                        info="Where to save processed images"
                    )
                with gr.Column():
                    proc_use_fp16 = gr.Checkbox(
                        label="Use FP16 Optimization (CUDA only)",
                        value=True,
                        info="Faster processing with NVIDIA GPUs - ignored on Apple Silicon/CPU"
                    )

            with gr.Row():
                with gr.Column():
                    proc_contrast = gr.Slider(
                        minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                        label="Contrast Scale",
                        info="Adjust image contrast"
                    )
                with gr.Column():
                    proc_upscale = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                        label="Upscale Factor",
                        info="Resize images"
                    )

            with gr.Row():
                with gr.Column():
                    proc_patch_size = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64,
                        label="Patch Size"
                    )
                with gr.Column():
                    proc_overlap = gr.Slider(
                        minimum=0, maximum=128, value=64, step=8,
                        label="Patch Overlap"
                    )

            # Advanced export options
            gr.HTML("""
            <div style="background: #ecfdf5; border: 1px solid #10b981; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4 style="color: #065f46; margin-top: 0;">🎨 Advanced Export Options</h4>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    proc_export_elements = gr.Checkbox(
                        label="Extract Individual Elements",
                        value=True,
                        info="Extract individual pottery elements as separate high-res images"
                    )
                with gr.Column():
                    proc_export_svg = gr.Checkbox(
                        label="Export SVG Versions",
                        value=True,
                        info="Convert outputs to scalable vector graphics (requires potrace)"
                    )

            with gr.Row():
                proc_button = gr.Button("Start Batch Processing", variant="primary", size="lg")

            # Attribution popup (initially hidden)
            attribution_popup = gr.Markdown(visible=False, elem_id="attribution-popup")

            proc_output_text = gr.Markdown(label="Processing Results")
            proc_output_comparisons = gr.Gallery(
                label="Before & After Comparisons",
                show_label=True,
                elem_id="proc-gallery",
                columns=3,
                height="auto",
                object_fit="contain"
            )

            # Open folder button and hidden path storage
            with gr.Row():
                open_folder_button = gr.Button(
                    "📁 Open Output Folder",
                    variant="secondary",
                    visible=False,
                    elem_id="open-folder-btn"
                )
            folder_path_hidden = gr.Textbox(visible=False)

            # Event handlers for processing
            proc_model_dropdown.change(
                fn=on_model_select,
                inputs=proc_model_dropdown,
                outputs=[proc_model_path_hidden, proc_model_prompt_hidden]
            )

            proc_button.click(
                fn=run_gradio_processing,
                inputs=[
                    proc_input, proc_model_path_hidden, proc_model_prompt_hidden,
                    proc_output_dir, proc_use_fp16, proc_contrast,
                    proc_patch_size, proc_overlap, proc_upscale,
                    proc_export_elements, proc_export_svg
                ],
                outputs=[proc_output_text, proc_output_comparisons, attribution_popup, open_folder_button, folder_path_hidden],
                show_progress="full"
            )

            # Add click handler for open folder button
            open_folder_button.click(
                fn=open_folder,
                inputs=[folder_path_hidden],
                outputs=[]
            )

        # TAB 5: About & Disclaimer
        with gr.Tab("About & Disclaimer", elem_id="about-tab"):
            # Logo and version info
            gr.HTML(f"""
            <div style="display: flex; align-items: center; padding: 30px; background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); border-radius: 12px; margin: 20px 0; border: 1px solid #d1d5db;">
                <div style="margin-right: 25px;">
                    <img src="data:image/png;base64,{img_data}"
                        alt="PyPotteryInk Logo"
                        style="border-radius: 10px; width: 80px; height: 80px; object-fit: contain; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);"/>
                </div>
                <div>
                    <h1 style="color: #1f2937; font-size: 2.5em; margin: 0; font-weight: 700;">
                        PyPotteryInk
                    </h1>
                    <p style="color: #6b7280; font-size: 1.2em; margin: 10px 0 5px 0; font-weight: 500;">
                        AI-Powered Archaeological Pottery Enhancement
                    </p>
                    <p style="color: #9ca3af; font-size: 1em; margin: 0;">
                        Version 1.0 • August 2025
                    </p>
                </div>
            </div>
            """)

            # Disclosure requirement
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); border-left: 4px solid #f59e0b; padding: 25px; border-radius: 8px; margin: 25px 0;">
                <h3 style="color: #92400e; margin-top: 0; font-size: 1.4em;">📢 AI DISCLOSURE REQUIREMENT</h3>
                <p style="color: #78350f; margin-bottom: 15px; font-size: 16px; line-height: 1.6;">
                    You are using PyPotteryInk version <strong>1.0</strong>, a Generative AI tool for translating
                    archaeological pottery drawings into publication-ready illustrations.
                </p>

                <h4 style="color: #92400e; margin: 20px 0 10px 0;">When publishing or presenting results that use PyPotteryInk, please include:</h4>
                <ul style="color: #78350f; font-size: 15px; line-height: 1.5; margin-left: 20px;">
                    <li>The version of PyPotteryInk used</li>
                    <li>The specific model used (e.g., '10k Model' or '6h-MCG Model')</li>
                    <li>The number of images processed</li>
                </ul>

                <h4 style="color: #92400e; margin: 20px 0 10px 0;">Suggested citation format:</h4>
                <div style="background: #fef3c7; padding: 15px; border-radius: 6px; margin: 10px 0;">
                    <p style="color: #78350f; margin: 0; font-style: italic; font-size: 15px; line-height: 1.5;">
                        "This research utilized PyPotteryInk (version 1.0) for the AI-assisted
                        translation of [number] pottery drawings. PyPotteryInk is a generative AI tool
                        developed by Lorenzo Cardarelli (<a href="https://github.com/lrncrd/PyPotteryInk" style="color: #92400e; text-decoration: underline;">https://github.com/lrncrd/PyPotteryInk</a>)."
                    </p>
                </div>
            </div>
            """)


            # Contact and attribution
            gr.HTML("""
            <div style="background: #ede9fe; border-left: 4px solid #8b5cf6; padding: 25px; border-radius: 8px; margin: 25px 0;">
                <h3 style="color: #5b21b6; margin-top: 0; font-size: 1.3em;">👥 Attribution & Contact</h3>
                <p style="color: #6b46c1; margin-bottom: 15px; font-size: 16px; line-height: 1.6;">
                    <strong>Developed by:</strong> Lorenzo Cardarelli<br>
                    <strong>GitHub:</strong> <a href="https://github.com/lrncrd/PyPotteryInk" style="color: #5b21b6; text-decoration: underline;">https://github.com/lrncrd/PyPotteryInk</a><br>
                    <strong>Research Context:</strong> Archaeological pottery documentation and publication
                </p>

                <p style="color: #6b46c1; margin: 0; font-size: 15px; line-height: 1.5;">
                    For questions, issues, or contributions, please visit the GitHub repository or contact the developer through the project's official channels.
                </p>
            </div>
            """)

    # Informative footer
    gr.HTML("""
    <div style="margin-top: 30px; padding: 20px; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb; text-align: center;">
        <p style="margin: 0; font-size: 14px; color: #6b7280;">
            <strong>PyPotteryInk</strong> - Advanced AI tool for archaeological pottery drawing enhancement<br>
            <em>Experimental tool - Ensure you have rights to process uploaded images and models</em>
        </p>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    print("Starting PyPotteryInk Archaeological Pottery Enhancement Tool...")
    # Try to find an available port if 7860 is taken
    import socket

    def find_free_port(start_port=7860, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return None

    port = find_free_port()
    if port is None:
        print("❌ Could not find an available port. Please close other Gradio instances.")
        exit(1)

    print(f"🌐 Using port: {port}")

    demo.launch(
        debug=True,
        show_error=True,
        share=False,
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True
    )

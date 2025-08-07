import gradio as gr
import os
import shutil
from pathlib import Path

# Assumiamo che lo script precedente sia salvato in un file chiamato `processor.py`
# Qui importiamo le funzioni principali
# from processor import run_diagnostics, process_folder

# 🔴 ATTENZIONE: se non hai già salvato lo script in un modulo, devi farlo!
# Salva il codice fornito nel file `processor.py` nella stessa directory.

from ink import run_diagnostics, process_folder  # Assicurati che il file sia salvato

# Directory temporanee
TEMP_INPUT = "temp_input"
TEMP_OUTPUT = "temp_output"
TEMP_DIAGNOSTICS = "temp_diagnostics"

os.makedirs(TEMP_INPUT, exist_ok=True)
os.makedirs(TEMP_OUTPUT, exist_ok=True)
os.makedirs(TEMP_DIAGNOSTICS, exist_ok=True)

def clear_temp_dirs():
    """Pulizia delle directory temporanee all'avvio."""
    for folder in [TEMP_INPUT, TEMP_OUTPUT, TEMP_DIAGNOSTICS]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

clear_temp_dirs()

def run_gradio_diagnostics(input_images, model_path, prompt, patch_size, overlap, contrast_values_str):
    if not input_images:
        return "❌ Nessuna immagine caricata per la diagnostica.", None
    if not model_path or not os.path.exists(model_path):
        return "❌ Percorso del modello non valido.", None

    # Salva le immagini caricate in una cartella temporanea
    for img in input_images:
        shutil.copy(img.name, TEMP_INPUT)

    # Elabora i contrasti
    try:
        contrast_values = [float(x.strip()) for x in contrast_values_str.split(",") if x.strip()]
        if not contrast_values:
            contrast_values = [1.0]
    except:
        contrast_values = [1.0]

    # Esegui diagnostica
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
        return "❌ Diagnostica fallita: nessuna immagine valida trovata.", None

    # Restituisci i risultati
    diagnostic_images = []
    for file in sorted(os.listdir(TEMP_DIAGNOSTICS)):
        if file.endswith(".png") or file.endswith(".jpg"):
            diagnostic_images.append(os.path.join(TEMP_DIAGNOSTICS, file))

    result_text = "✅ Diagnostica completata! Visualizzazioni generate:"
    return result_text, diagnostic_images


def run_gradio_processing(input_images, model_path, prompt, output_dir, use_fp16, contrast_scale,
                          patch_size, overlap, upscale):
    if not input_images:
        return "❌ Nessuna immagine da elaborare.", None, None
    if not model_path or not os.path.exists(model_path):
        return "❌ Percorso del modello non valido.", None, None

    # Pulisci e prepara le cartelle
    shutil.rmtree(TEMP_INPUT, ignore_errors=True)
    os.makedirs(TEMP_INPUT, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Copia immagini caricate
    for img in input_images:
        shutil.copy(img.name, TEMP_INPUT)

    # Esegui elaborazione batch
    try:
        results = process_folder(
            input_folder=TEMP_INPUT,
            model_path=model_path,
            prompt=prompt,
            output_dir=output_dir,
            use_fp16=use_fp16,
            contrast_scale=contrast_scale,
            patch_size=patch_size,
            overlap=overlap,
            upscale=upscale
        )

        # Genera zip della cartella output
        shutil.make_archive("processed_images", 'zip', output_dir)

        summary = (
            f"✅ Elaborazione completata!\n"
            f"• Successo: {results['successful']}\n"
            f"• Falliti: {results['failed']}\n"
            f"• Tempo medio: {results['average_time']:.2f}s\n"
            f"• Log: {results['log_file']}"
        )
        return summary, "processed_images.zip", results["comparison_dir"]
    except Exception as e:
        return f"❌ Errore durante l'elaborazione: {str(e)}", None, None


# --- INTERFACCIA GRADIO ---
with gr.Blocks(title="🖼️ Pix2Pix_Turbo Image Enhancer") as demo:
    gr.Markdown("""
    # 🚀 Pix2Pix_Turbo - Image Enhancement Tool
    Carica immagini e applica miglioramenti con un modello AI basato su patch.
    """)

    with gr.Tabs():
        # TAB 1: Diagnostica
        with gr.Tab("🔍 Diagnostica"):
            gr.Markdown("Esegui test preliminari: visualizzazione patch e confronto contrasto.")
            with gr.Row():
                diag_input = gr.File(file_count="multiple", label="Carica immagini per diagnostica", type="filepath")
            with gr.Row():
                diag_model = gr.Textbox(label="Percorso del modello (file .pth)", value="./6h-MC.pkl")
                diag_prompt = gr.Textbox(label="Prompt", value="make it ready for publication")
            with gr.Row():
                diag_patch_size = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Patch Size")
                diag_overlap = gr.Slider(minimum=0, maximum=128, value=64, step=8, label="Overlap")
            with gr.Row():
                diag_contrast = gr.Textbox(
                    label="Valori di contrasto (separati da virgola)",
                    value="0.75, 1.0, 1.5, 2.0"
                )
            diag_button = gr.Button("📊 Esegui Diagnostica")
            diag_output_text = gr.Textbox(label="Risultato")
            diag_output_images = gr.Gallery(label="Immagini di Diagnostica")

            diag_button.click(
                fn=run_gradio_diagnostics,
                inputs=[diag_input, diag_model, diag_prompt, diag_patch_size, diag_overlap, diag_contrast],
                outputs=[diag_output_text, diag_output_images]
            )

        # TAB 2: Elaborazione
        with gr.Tab("⚙️ Elaborazione Batch"):
            gr.Markdown("Elabora un batch di immagini con il modello.")
            with gr.Row():
                proc_input = gr.File(file_count="multiple", label="Carica immagini da elaborare", type="filepath")
            with gr.Row():
                proc_model = gr.Textbox(label="Percorso del modello", value="./models/pix2pix_turbo_canny.pth")
                proc_prompt = gr.Textbox(label="Prompt", value="make it ready for publication")
            with gr.Row():
                proc_output_dir = gr.Textbox(label="Cartella di output", value="./output")
                proc_use_fp16 = gr.Checkbox(label="Usa FP16 (più veloce, meno memoria)")
            with gr.Row():
                proc_contrast = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="Scala del contrasto")
                proc_upscale = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Upscale (1.0 = nessun cambio)")
            with gr.Row():
                proc_patch_size = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Patch Size")
                proc_overlap = gr.Slider(minimum=0, maximum=128, value=64, step=8, label="Overlap")
            proc_button = gr.Button("🚀 Avvia Elaborazione")
            proc_output_text = gr.Textbox(label="Riepilogo")
            proc_output_zip = gr.File(label="Scarica risultati (ZIP)")
            proc_output_comparisons = gr.Gallery(label="Confronti Originale vs Processato")

            proc_button.click(
                fn=run_gradio_processing,
                inputs=[
                    proc_input, proc_model, proc_prompt, proc_output_dir, proc_use_fp16,
                    proc_contrast, proc_patch_size, proc_overlap, proc_upscale
                ],
                outputs=[proc_output_text, proc_output_zip, proc_output_comparisons]
            )

    gr.Markdown("""
    ---
    🔐 **Nota:** Questo strumento è a scopo sperimentale. Assicurati di avere i diritti sulle immagini e sul modello.
    """)

# Lancia l'app
if __name__ == "__main__":
    demo.launch(debug=True)

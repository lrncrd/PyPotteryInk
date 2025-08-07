import gradio as gr
import os
import shutil
from pathlib import Path
from hardware_check import run_hardware_check
import requests
from pathlib import Path
# Assumiamo che lo script precedente sia salvato in un file chiamato `processor.py`
# Qui importiamo le funzioni principali
# from processor import run_diagnostics, process_folder

# 🔴 ATTENZIONE: se non hai già salvato lo script in un modulo, devi farlo!
# Salva il codice fornito nel file `processor.py` nella stessa directory.

from ink import run_diagnostics, process_folder  # Assicurati che il file sia salvato

# Configurazione modelli
MODELS = {
    "10k Model": {
        "description": "General-purpose model for pottery drawings",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/model_10k.pkl?download=true",
        "filename": "model_10k.pkl"
    },
    "6h-MCG Model": {
        "description": "High-quality model for Bronze Age drawings",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MCG.pkl?download=true",
        "filename": "6h-MCG.pkl"
    },
    "6h-MC Model": {
        "description": "High-quality model for Protohistoric and Historic drawings",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MC.pkl?download=true",
        "filename": "6h-MC.pkl"
    },
    "4h-PAINT Model": {
        "description": "Tailored model for Historic and painted pottery",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/4h-PAINT.pkl?download=true",
        "filename": "4h-PAINT.pkl"
    }
}

# Crea la cartella models se non esiste
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def download_model(model_name):
    """Scarica il modello selezionato se non esiste già"""
    model_info = MODELS[model_name]
    model_path = os.path.join(MODELS_DIR, model_info["filename"])

    if not os.path.exists(model_path):
        try:
            print(f"Downloading {model_name}...")
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"{model_name} downloaded successfully!")
            return model_path
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
            return None
    else:
        print(f"{model_name} already exists, skipping download")
        return model_path

def get_model_dropdown():
    """Crea la descrizione per il dropdown"""
    choices = []
    for name, info in MODELS.items():
        choices.append(f"{name} - {info['description']}")
    return choices

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
        # shutil.make_archive("processed_images", 'zip', output_dir)

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


def run_hardware_check():
    """Funzione wrapper per il check hardware"""
    checker = HardwareChecker()
    return checker.generate_report()


# --- INTERFACCIA GRADIO ---
with gr.Blocks(title="🖼️ Pix2Pix_Turbo Image Enhancer") as demo:
    gr.Markdown("""
    # 🚀 Pix2Pix_Turbo - Image Enhancement Tool
    Carica immagini e applica miglioramenti con un modello AI basato su patch.
    """)

    with gr.Tabs():
        # TAB 1: Hardware
        with gr.Tab("🛠️ Verifica Hardware"):
            gr.Markdown("""
            ## Verifica le specifiche del tuo computer
            Questo strumento richiede risorse significative. Verifica che il tuo hardware sia adeguato.
            """)
            hw_btn = gr.Button("🔍 Analizza Hardware", variant="primary")
            hw_report = gr.Markdown()
            hw_btn.click(fn=run_hardware_check, outputs=hw_report)

            gr.Markdown("""
            ### Requisiti consigliati:
            - **GPU:** NVIDIA con almeno 8GB VRAM (minimo 4GB)
            - **CPU:** 4+ core moderni
            - **RAM:** 16GB (minimo 8GB)
            - **Disco:** SSD veloce (NVMe consigliato)

            ### Note importanti:
            - L'uso della GPU è fondamentale per prestazioni accettabili
            - Su portatili, assicurarsi di:
              - Usare l'alimentazione collegata
              - Avere una buona ventilazione
              - Considerare una base di raffreddamento
            """)

        # TAB 2: Diagnostica
        with gr.Tab("🔍 Diagnostica"):
            gr.Markdown("Esegui test preliminari: visualizzazione patch e confronto contrasto.")
            with gr.Row():
                diag_input = gr.File(file_count="multiple", label="Carica immagini per diagnostica", type="filepath")
            with gr.Row():
                # diag_model = gr.Textbox(label="Percorso del modello (file .pth)", value="./6h-MC.pkl")
                model_dropdown = gr.Dropdown(
                    label="Seleziona modello",
                    choices=get_model_dropdown(),
                    value="6h-MC Model - High-quality model for Protohistoric and Historic drawings (38.3MB)"
                )
                diag_prompt = gr.Textbox(label="Prompt", value="make it ready for publication")
            # Aggiungi questo componente nascosto per il percorso del modello
            model_path_hidden = gr.Textbox(visible=False)
            # Quando si seleziona un modello, scaricalo se necessario
            def on_model_select(selection):
                # Estrai il nome del modello dalla selezione
                model_name = selection.split(" - ")[0]
                path = download_model(model_name)
                return path if path else ""

            model_dropdown.change(
                fn=on_model_select,
                inputs=model_dropdown,
                outputs=model_path_hidden
            )

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
                inputs=[diag_input, model_path_hidden, diag_prompt, diag_patch_size, diag_overlap, diag_contrast],
                outputs=[diag_output_text, diag_output_images]
            )

        # TAB 3: Elaborazione
        with gr.Tab("⚙️ Elaborazione Batch"):
            gr.Markdown("Elabora un batch di immagini con il modello.")
            with gr.Row():
                proc_input = gr.File(file_count="multiple", label="Carica immagini da elaborare", type="filepath")
            with gr.Row():
                # proc_model = gr.Textbox(label="Percorso del modello", value="./models/pix2pix_turbo_canny.pth")
                proc_model_dropdown = gr.Dropdown(
                    label="Seleziona modello",
                    choices=get_model_dropdown(),
                    value="6h-MC Model - High-quality model for Protohistoric and Historic drawings (38.3MB)"
                )
                proc_prompt = gr.Textbox(label="Prompt", value="make it ready for publication")
                proc_model_path_hidden = gr.Textbox(visible=False)
                proc_model_dropdown.change(
                    fn=on_model_select,
                    inputs=proc_model_dropdown,
                    outputs=proc_model_path_hidden
                )

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
                    proc_input, proc_model_path_hidden, proc_prompt, proc_output_dir, proc_use_fp16,
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

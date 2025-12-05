# PyPotteryInk

<div align="center">

<img src="imgs/LogoInk.png" width="250"/>

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://lrncrd.github.io/PyPotteryInk/)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://lrncrd.github.io/PyPotteryInk/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Models-PyPotteryInk-yellow.svg)](https://huggingface.co/lrncrd/PyPotteryInk)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.culher.2025.06.016-blue.svg)](https://doi.org/10.1016/j.culher.2025.06.016)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%20%7C%20MPS-green.svg)](https://github.com/lrncrd/PyPotteryInk)



*Transform archaeological pottery drawings into publication-ready illustrations with AI*

🚀 Try the [demo](https://githubtocolab.com/lrncrd/PyPotteryInk/blob/main/PyPotteryInk_example.ipynb) on Google Colab 🚀

---

### ☕ Support This Project

If you find PyPotteryInk useful for your research, consider supporting its development:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/lrncrd)

Your support helps maintain and improve this open-source tool for the archaeological community!


</div>


## 🎯 Overview

As part of the **PyPottery** toolkit, `PyPotteryInk` is a deep learning application for automating the digital inking process of archaeological pottery drawings. It transforms pencil drawings into publication-ready illustrations while preserving the original characteristics and enhancing their analytical power.

<div align="center">
<img src="imgs/comparison.jpg" width="800"/>
<p><em>Example of PyPotteryInk transformation: from pencil sketch to publication-ready illustration</em></p>
</div>

## ✨ Features

- 🚀 **Single-Step Translation**: Convert pencil drawings to inked versions using state-of-the-art diffusion models
- 🖼️ **High-Resolution Processing**: Advanced patch-based system for handling large drawings
- 🎨 **Stippling Control**: Fine-grained control over dot patterns and shading
- 📂 **Batch Processing**: Efficient handling of multiple drawings
- 🖥️ **Multi-GPU Support**: Now supports CUDA (NVIDIA), MPS (Apple Silicon M1/M2/M3), and CPU fallback
- 🌐 **Web Interface**: User-friendly Flask-based web interface for easy access
- 📊 **Preprocessing Tools**: Built-in image analysis and optimization with detailed statistics visualization
- 🔧 **Easy Installation**: One-click installation scripts for all major operating systems
- 🎯 **Custom Models**: Support for uploading and using custom-trained models

## 🚀 Quick Start

### Installation

PyPotteryInk includes a unified installation script that works on all platforms:

```bash
git clone https://github.com/lrncrd/PyPotteryInk.git
cd PyPotteryInk
python install.py
```

The installation script will:
- Create a virtual environment
- Install all dependencies
- Download required models
- Set up the application

### Running the Application

After installation:
- **Windows**: Double-click `PyPotteryInk_WIN.bat` or run `python app.py` from terminal
- **macOS/Linux**: Run `./PyPotteryInk_UNIX.sh` or `python app.py`

The web interface will open automatically in your browser at `http://127.0.0.1:5003`.

### Web Interface

Version 2.0.0 introduces a modern Flask-based web interface with real-time processing updates.

<div align="center">
<img src="imgs/gui_example.png" width="800"/>
</div>

1. **Hardware Check Tab**: Verify your system meets requirements
2. **Model Management**: Download and manage AI models
3. **Model Diagnostics Tab**: Test different settings before processing
4. **Preprocessing Tab**: 
   - Calculate detailed statistics from your dataset
   - View distribution plots and summary tables
   - Apply optimizations based on statistical analysis
5. **Batch Processing Tab**: Process multiple images with real-time progress tracking
6. **Custom Model Upload**: Upload and use your own fine-tuned models

## 📚 Documentation

<div align="center">
  <a href="https://lrncrd.github.io/PyPotteryInk/">
    <img src="https://img.shields.io/badge/📖%20Read%20the%20Docs-PyPotteryInk-4A5568?style=for-the-badge&logo=read-the-docs&logoColor=white" alt="Documentation"/>
  </a>
</div>

## 🤖 Available Models

| Model | Description | Checkpoint Size | Link |
|-------|-------------|------|------|
| **10k Model** | General-purpose model for pottery drawings | 38.3MB | [Download](https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/model_10k.pkl?download=true) |
| **6h-MCG Model** | High-quality model for Bronze Age drawings | 38.3MB | [Download](https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MCG.pkl?download=true) |
| **6h-MC Model** | High-quality model for Protohistoric and Historic drawings | 38.3MB | [Download](https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/6h-MC.pkl?download=true) |
| **4h-PAINT Model** | Tailored model for Historic and painted pottery | 38.3MB | [Download](https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/4h-PAINT.pkl?download=true) |
| **5h-PAPERGRID Model** | Tailored model for handling paper grid tables (does not support shadows) | 38.3MB | [Download](https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/5h_PAPERGRID.pkl?download=true) |



All models support custom fine-tuning for specific archaeological contexts or styles.

## ⚡ Benchmarks

| GPU                 | Processing time for `test_image.jpg` (s) | FP16 Support |
| ------------------- | -------------------------------- | ------------ |
| 3070Ti (Windows 11) | ~50-55                     | ✅ Yes        |
| T4 (Google Colab)   | ~55-60                           | ✅ Yes        |
| M2 Pro (macOS)      | ~65-75                          | ❌ No (FP32)  |
| M1 (macOS)          | ~80-90                          | ❌ No (FP32)  |
| CPU (i7-9700K)     | ~300-400                        | ❌ No (FP32)  |

**Note**: FP16 (half precision) is only supported on CUDA GPUs. Apple Silicon (MPS) and CPU use FP32 for stability. 


## 🖥️ System Requirements

### Minimum Requirements
- **CPU**: 4+ cores (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB (16GB recommended)
- **GPU**: 
  - NVIDIA: GTX 1060 6GB or better (RTX series recommended)
  - Apple: M1/M2/M3 with 8GB+ unified memory
- **Storage**: 5GB free space for models and processing
- **OS**: Windows 10/11, macOS 11+, Linux (Ubuntu 20.04+)

### GPU Support
- **CUDA (NVIDIA)**: Full support with FP16 optimization
- **MPS (Apple Silicon)**: Full support with FP32 (automatic)
- **CPU**: Fallback mode (significantly slower)



## 📢 AI Disclosure and Citation

PyPotteryInk uses Generative AI to translate archaeological pottery drawings into publication-ready illustrations. To promote transparency about the use of Generative AI and proper attribution in scientific research, is required all users to include the following disclosure statement in any publication, presentation, or report that utilizes PyPotteryInk:

```
This research utilized PyPotteryInk (version 2.0) for the AI-assisted translation of [number] pottery drawings. PyPotteryInk is a generative AI tool developed by Lorenzo Cardarelli (https://github.com/lrncrd/PyPotteryInk).
```

Where you need to specify the software version and the number of processed pottery drawings.

### Usage Reporting

By using PyPotteryInk, you agree to:

1. Clearly indicate in your methods section which model was used (e.g., "10k Model", "6h-MCG Model" or a custom model)
2. Specify the number of images processed with PyPotteryInk
3. Include the version number of PyPotteryInk used in your research

### Citation

If you use PyPotteryInk in your research, please cite:

```bibtex
@software{cardarelli2025pypotteryink,
  author = {Cardarelli, Lorenzo},
  title = {PyPotteryInk: Transform archaeological pottery drawings into publication-ready illustrations with AI},
  year = {2025},
  url = {https://github.com/lrncrd/PyPotteryInk},
  version = {2.0}
}
```

or 

```bibtex
@article{cardarelli_pypotteryink_2025,
	title = {{PyPotteryInk}: One-step diffusion model for sketch to publication-ready archaeological drawings},
	volume = {74},
	issn = {1296-2074},
	url = {https://www.sciencedirect.com/science/article/pii/S1296207425001268},
	doi = {10.1016/j.culher.2025.06.016},
	shorttitle = {{PyPotteryInk}},
	pages = {300--310},
	journaltitle = {Journal of Cultural Heritage},
	author = {Cardarelli, Lorenzo},
	date = {2025-07-01},
	keywords = {Archaeological drawing, Diffusion models, Generative {AI}, Image-to-image translation, Pottery},
}
```


## 👥 Contributors

<a href="https://github.com/lrncrd/PyPotteryInk/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=lrncrd/PyPotteryInk" />
</a>



Developed with ❤️ by [Lorenzo Cardarelli](https://github.com/lrncrd)

Based on img2img-turbo by [GaParmar](https://github.com/GaParmar/img2img-turbo)

The original code was released under the MIT Licence. The changes made in this fork are released under the Apache License 2.0.

## 🆕 What's New in Version 2.0.0

- **Flask Web Interface**: Complete redesign with Flask for better performance and reliability
- **Real-time Progress Updates**: Server-Sent Events (SSE) for live processing status
- **Custom Model Support**: Upload and use your own fine-tuned models
- **5h-PAPERGRID Model**: New specialized model for handling paper grid tables
- **Improved Session Management**: Better handling of multiple processing sessions
- **Better Error Handling**: More informative error messages and recovery options
- **Streamlined Interface**: Cleaner, more intuitive user experience
- **Directory Picker**: Native file system dialogs for selecting output directories
- **Comparison Images**: Automatic generation of before/after comparisons

## 🛠️ Development Setup

For developers who want to contribute:

```bash
git clone https://github.com/lrncrd/PyPotteryInk.git
cd PyPotteryInk
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🐛 Troubleshooting

### Common Issues


1. **Out of memory errors**
   - Reduce patch size in settings (try 384 or 256)
   - Close other applications
   - Use CPU mode as fallback (slower but more stable)

2. **Models not downloading**
   - Check your internet connection
   - Models are downloaded from Hugging Face (may be blocked in some regions)
   - Manual download links are available in the table above

3. **Flask server not starting**
   - Check if port 5003 is already in use
   - Try closing other applications that might use this port
   - Alternatively, modify the port in `app.py` (last line)

4. **Statistics visualization not showing**
   - Ensure scipy is installed: `pip install scipy`
   - Check that you have at least 2 images for meaningful statistics
   - Verify the "Generate visualization plots" checkbox is enabled

5. **Custom model not working**
   - Ensure the model file is in `.pkl` format
   - Check that the model is compatible with the PyPotteryInk architecture
   - Verify the file was uploaded successfully before processing

---

# PyPotteryInk

<div align="center">

<img src="imgs/pypotteryink.png" width="250"/>

[![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://lrncrd.github.io/PyPotteryInk/)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://lrncrd.github.io/PyPotteryInk/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Models-PyPotteryInk-yellow.svg)](https://huggingface.co/lrncrd/PyPotteryInk)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

*Transform archaeological pottery drawings into publication-ready illustrations with AI*

🚀 Try the [demo](https://githubtocolab.com/lrncrd/PyPotteryInk/blob/main/PyPotteryInk_example.ipynb) on Google Colab 🚀


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



All models support custom fine-tuning for specific archaeological contexts or styles.

## ⚡ Benchmarks


| GPU                 | Mean processing time for $512 \cdot 512$ pixel patch (s) |
| ------------------- | -------------------------------- |
| 3070Ti (Windows 11) | 6.62                             |
| 3060 (Windows 11)   | 21.29                            |
| 3060 (WSL Ubuntu)   | 20.89                            |
| T4 (Google Colab)   | 0.56                             |

The benchmarks were performed using the `test.py` script. 

## 📢 AI Disclosure and Citation

PyPotteryInk uses Generative AI to translate archaeological pottery drawings into publication-ready illustrations. To promote transparency about the use of Generative AI and proper attribution in scientific research, is required all users to include the following disclosure statement in any publication, presentation, or report that utilizes PyPotteryInk:

```
This research utilized PyPotteryInk (version X.X.X) for the AI-assisted translation of [number] pottery drawings. PyPotteryInk is a generative AI tool developed by Lorenzo Cardarelli (https://github.com/lrncrd/PyPotteryInk).
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
  version = {0.0.1}
}
```


## 👥 Contributors

<div align="center">
<a href="https://github.com/lrncrd">
  <img src="https://github.com/lrncrd.png" width="50px" alt="Lorenzo Cardarelli" style="border-radius: 50%"/>
</a>



Developed with ❤️ by [Lorenzo Cardarelli](https://github.com/lrncrd)
</div>

Based on img2img-turbo by [GaParmar](https://github.com/GaParmar/img2img-turbo)

The original code was released under the MIT Licence. The changes made in this fork are released under the Apache License 2.0.

---

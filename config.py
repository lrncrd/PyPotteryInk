# PyPotteryInk Flask Configuration

# Server Configuration
HOST = '127.0.0.1'
PORT = 5001
DEBUG = True

# Folders Configuration
MODELS_DIR = 'models'
UPLOAD_FOLDER = 'temp_uploads'
OUTPUT_FOLDER = 'temp_output'
DIAGNOSTICS_FOLDER = 'temp_diagnostics'

# Upload Configuration
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max upload size
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# Processing Configuration
DEFAULT_PATCH_SIZE = 512
DEFAULT_OVERLAP = 64
DEFAULT_CONTRAST_SCALE = 1.0
DEFAULT_UPSCALE = 1

# Model Configuration
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
    },
    "5h-PAPERGRID Model": {
        "description": "Tailored model for handling paper grid tables (DO NOT SUPPORT SHADOWS)",
        "size": "38.3MB",
        "url": "https://huggingface.co/lrncrd/PyPotteryInk/resolve/main/5h_PAPERGRID.pkl?download=true",
        "filename": "5h_PAPERGRID.pkl",
        "prompt": "enhance pottery drawing for publication"
    }
}

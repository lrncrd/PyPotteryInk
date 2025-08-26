#!/bin/bash
echo "Starting PyPotteryInk Professional Interface..."
source venv/bin/activate

# Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run the app
echo "Loading interface..."
python app.py

import sys
import threading
from contextlib import contextmanager

class ProgressCapture:
    """Capture and redirect console output including tqdm progress bars"""
    
    def __init__(self, callback=None):
        self.callback = callback
        self.buffer = []
        self._lock = threading.Lock()
        
    def write(self, text):
        """Capture text output"""
        if text and text.strip():
            with self._lock:
                # Handle tqdm progress bars
                if '\r' in text:
                    # Overwrite last line for progress bars
                    if self.buffer and self.buffer[-1].startswith('\r'):
                        self.buffer[-1] = text.strip()
                    else:
                        self.buffer.append(text.strip())
                else:
                    self.buffer.append(text.strip())
                
                if self.callback:
                    self.callback(self.get_full_output())
    
    def flush(self):
        pass
    
    def get_full_output(self):
        """Get all captured output as a single string"""
        with self._lock:
            return '\n'.join(self.buffer)
    
    def get_last_n_lines(self, n=30):
        """Get last n lines of output"""
        with self._lock:
            return '\n'.join(self.buffer[-n:])

@contextmanager
def capture_output(callback=None):
    """Context manager to capture stdout and stderr"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    capture = ProgressCapture(callback)
    sys.stdout = capture
    sys.stderr = capture
    
    try:
        yield capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def create_progress_callback(status_updater, yield_func=None):
    """Create a callback that updates status and optionally yields"""
    def callback(full_text):
        # Update status
        lines = full_text.split('\n')
        # Keep last 25 lines for display
        display_text = '\n'.join(lines[-25:])
        status_updater(display_text)
        
        # If we have a yield function, call it
        if yield_func:
            yield_func()
    
    return callback
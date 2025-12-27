# hardware_check.py
import torch
import platform
import psutil
import time
import os
from typing import Dict, List, Tuple, Any

try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

# Status constants
STATUS_EXCELLENT = "excellent"
STATUS_ADEQUATE = "adequate"
STATUS_LIMITED = "limited"

# Suitability levels
SUITABILITY_HIGHLY_RECOMMENDED = "highly_recommended"
SUITABILITY_RECOMMENDED = "recommended"
SUITABILITY_LIMITED_USE = "limited_use"
SUITABILITY_NOT_RECOMMENDED = "not_recommended"

class HardwareChecker:
    """Internal class for hardware checking"""
    def __init__(self):
        self.info = self.get_hardware_info()
        self.components = {}
        self.evaluate_hardware()

    def get_hardware_info(self) -> Dict[str, Any]:
        """Collects system hardware information"""
        info = {}

        # GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["gpu"] = f"{gpu_name} ({gpu_mem:.1f} GB)"
            info["gpu_mem"] = gpu_mem
            info["gpu_name"] = gpu_name.lower()
            info["has_cuda"] = True
            info["has_mps"] = False
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # Apple Metal Performance Shaders
            info["gpu"] = "Apple Metal Performance Shaders (MPS)"
            info["gpu_mem"] = psutil.virtual_memory().total / (1024**3)  # Uses total RAM as reference
            info["gpu_name"] = "mps"
            info["has_mps"] = True
            info["has_cuda"] = False
        else:
            info["gpu"] = "No accelerated GPU detected"
            info["gpu_mem"] = 0
            info["gpu_name"] = ""
            info["has_cuda"] = False
            info["has_mps"] = False

        # CPU information
        if CPUINFO_AVAILABLE:
            cpu_info = cpuinfo.get_cpu_info()
            info["cpu"] = f"{cpu_info['brand_raw']} ({psutil.cpu_count()} cores)"
            info["cpu_brand"] = cpu_info['brand_raw'].lower()
        else:
            info["cpu"] = f"{platform.processor()} ({psutil.cpu_count()} cores)"
            info["cpu_brand"] = platform.processor().lower()
        info["cpu_cores"] = psutil.cpu_count()

        # RAM information
        ram = psutil.virtual_memory().total / (1024**3)
        info["ram"] = f"{ram:.1f} GB"
        info["ram_gb"] = ram

        # Disk speed test
        disk_speed, disk_speed_values = self.test_disk_speed()
        info["disk_speed"] = disk_speed
        info["disk_speed_read"] = disk_speed_values[0]
        info["disk_speed_write"] = disk_speed_values[1]

        # Operating system
        info["os"] = platform.system().lower()
        
        return info

    def test_disk_speed(self) -> Tuple[str, Tuple[float, float]]:
        """Tests disk read/write speed"""
        try:
            test_file = "speed_test.tmp"
            start_time = time.time()

            # Write
            with open(test_file, "wb") as f:
                f.write(os.urandom(100 * 1024 * 1024))  # 100MB
            write_time = time.time() - start_time

            # Read
            start_time = time.time()
            with open(test_file, "rb") as f:
                _ = f.read()
            read_time = time.time() - start_time

            os.remove(test_file)

            write_speed = (100 / write_time)  # MB/s
            read_speed = (100 / read_time)    # MB/s
            return f"Read: {read_speed:.0f} MB/s | Write: {write_speed:.0f} MB/s", (read_speed, write_speed)
        except Exception as e:
            return f"Test failed: {str(e)}", (0, 0)

    def evaluate_hardware(self):
        """Evaluates hardware and generates categorical ratings"""
        
        # GPU evaluation
        gpu_status = STATUS_LIMITED
        gpu_message = ""
        
        if self.info.get("has_mps", False):
            gpu_status = STATUS_EXCELLENT
            gpu_message = "Apple Silicon with Metal acceleration"
        elif self.info.get("has_cuda", False):
            if "rtx" in self.info["gpu_name"] and self.info["gpu_mem"] >= 6:
                gpu_status = STATUS_EXCELLENT
                gpu_message = f"RTX GPU with {self.info['gpu_mem']:.0f}GB VRAM"
            elif self.info["gpu_mem"] >= 8:
                gpu_status = STATUS_EXCELLENT
                gpu_message = f"GPU with {self.info['gpu_mem']:.0f}GB VRAM"
            elif self.info["gpu_mem"] >= 4:
                gpu_status = STATUS_ADEQUATE
                gpu_message = f"GPU with {self.info['gpu_mem']:.0f}GB VRAM (may have limitations with large images)"
            else:
                gpu_status = STATUS_LIMITED
                gpu_message = f"GPU with only {self.info['gpu_mem']:.0f}GB VRAM"
        else:
            gpu_status = STATUS_LIMITED
            gpu_message = "No GPU acceleration available (CPU only)"
        
        self.components["gpu"] = {
            "name": "GPU",
            "icon": "🎮",
            "value": self.info["gpu"],
            "status": gpu_status,
            "message": gpu_message
        }

        # RAM evaluation
        ram_gb = self.info["ram_gb"]
        if ram_gb >= 16:
            ram_status = STATUS_EXCELLENT
            ram_message = "Ideal for large image batches"
        elif ram_gb >= 8:
            ram_status = STATUS_ADEQUATE
            ram_message = "Adequate for processing"
        else:
            ram_status = STATUS_LIMITED
            ram_message = "May have issues with batch processing"
        
        self.components["ram"] = {
            "name": "RAM",
            "icon": "💾",
            "value": self.info["ram"],
            "status": ram_status,
            "message": ram_message
        }

        # CPU evaluation
        cpu_cores = self.info["cpu_cores"]
        cpu_brand = self.info["cpu_brand"]
        
        # Check for Apple Silicon
        apple_silicon = "apple" in cpu_brand or "m1" in cpu_brand or "m2" in cpu_brand or "m3" in cpu_brand or "m4" in cpu_brand
        
        # Check for modern CPUs
        modern_cpu = any(brand in cpu_brand for brand in ["intel", "amd"]) and \
                     any(gen in cpu_brand for gen in ["i5", "i7", "i9", "ryzen 5", "ryzen 7", "ryzen 9"])
        
        if apple_silicon:
            cpu_status = STATUS_EXCELLENT
            cpu_message = "Apple Silicon processor"
        elif modern_cpu and cpu_cores >= 8:
            cpu_status = STATUS_EXCELLENT
            cpu_message = f"Modern processor with {cpu_cores} cores"
        elif modern_cpu and cpu_cores >= 4:
            cpu_status = STATUS_ADEQUATE
            cpu_message = f"Modern processor with {cpu_cores} cores"
        elif cpu_cores >= 4:
            cpu_status = STATUS_ADEQUATE
            cpu_message = f"Processor with {cpu_cores} cores"
        else:
            cpu_status = STATUS_LIMITED
            cpu_message = f"Only {cpu_cores} cores available"
        
        self.components["cpu"] = {
            "name": "CPU",
            "icon": "🖥️",
            "value": self.info["cpu"],
            "status": cpu_status,
            "message": cpu_message
        }

        # Disk evaluation
        read_speed = self.info["disk_speed_read"]
        write_speed = self.info["disk_speed_write"]
        
        if read_speed >= 500 and write_speed >= 300:
            disk_status = STATUS_EXCELLENT
            disk_message = "Fast NVMe SSD"
        elif read_speed >= 200 and write_speed >= 100:
            disk_status = STATUS_ADEQUATE
            disk_message = "SSD storage"
        else:
            disk_status = STATUS_LIMITED
            disk_message = "Slow storage (consider upgrading to SSD)"
        
        self.components["disk"] = {
            "name": "Storage",
            "icon": "💿",
            "value": self.info["disk_speed"],
            "status": disk_status,
            "message": disk_message
        }

    def get_suitability(self) -> Tuple[str, str, str]:
        """Returns overall suitability level, label and emoji"""
        gpu_status = self.components["gpu"]["status"]
        
        # Count status levels
        statuses = [c["status"] for c in self.components.values()]
        has_limited = STATUS_LIMITED in statuses
        all_excellent = all(s == STATUS_EXCELLENT for s in statuses)
        
        if gpu_status == STATUS_LIMITED:
            return SUITABILITY_NOT_RECOMMENDED, "Not Recommended", "🔴"
        elif has_limited:
            return SUITABILITY_LIMITED_USE, "Limited Use", "🟡"
        elif all_excellent or gpu_status == STATUS_EXCELLENT:
            return SUITABILITY_HIGHLY_RECOMMENDED, "Highly Recommended", "🟢"
        else:
            return SUITABILITY_RECOMMENDED, "Recommended", "🟠"

    def get_structured_report(self) -> Dict[str, Any]:
        """Returns structured report as dictionary for JSON serialization"""
        suitability_level, suitability_label, suitability_emoji = self.get_suitability()
        
        # Generate conclusion message
        if suitability_level == SUITABILITY_HIGHLY_RECOMMENDED:
            conclusion = "Your hardware is excellent for PyPotteryInk!"
        elif suitability_level == SUITABILITY_RECOMMENDED:
            conclusion = "Your hardware is adequate for PyPotteryInk."
        elif suitability_level == SUITABILITY_LIMITED_USE:
            conclusion = "Your hardware can run PyPotteryInk but may have performance limitations."
        else:
            conclusion = "Your hardware is not recommended. Processing may be very slow."
        
        return {
            "suitability": {
                "level": suitability_level,
                "label": suitability_label,
                "emoji": suitability_emoji,
                "conclusion": conclusion
            },
            "components": self.components,
            "tips": [
                "Keep laptop well ventilated and connected to power during processing",
                "Close other applications to maximize available memory"
            ]
        }

    def generate_report(self) -> str:
        """Generates a complete report in markdown (legacy support)"""
        structured = self.get_structured_report()
        
        report = f"## {structured['suitability']['emoji']} {structured['suitability']['label']}\n\n"
        report += f"{structured['suitability']['conclusion']}\n\n"
        
        report += "## 🖥️ Hardware Specifications\n"
        for key, comp in structured['components'].items():
            status_icon = "✅" if comp['status'] == STATUS_EXCELLENT else "⚠️" if comp['status'] == STATUS_ADEQUATE else "❌"
            report += f"- **{comp['name']}:** {comp['value']} {status_icon}\n"
        
        return report


def run_hardware_check() -> str:
    """
    Public function to run hardware check
    Returns a formatted markdown report
    """
    checker = HardwareChecker()
    return checker.generate_report()
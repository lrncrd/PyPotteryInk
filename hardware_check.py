# hardware_check.py
import torch
import platform
import psutil
import time
import os
from typing import Dict, List, Tuple

try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

class HardwareChecker:
    """Internal class for hardware checking"""
    def __init__(self):
        self.info = self.get_hardware_info()
        self.recommendations = []
        self.warnings = []
        self.score = 0
        self.max_score = 100
        self.evaluate_hardware()

    def get_hardware_info(self) -> Dict[str, str]:
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
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # Apple Metal Performance Shaders
            info["gpu"] = "Apple Metal Performance Shaders (MPS)"
            info["gpu_mem"] = psutil.virtual_memory().total / (1024**3)  # Usa RAM totale come riferimento
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
            return f"Read: {read_speed:.1f} MB/s | Write: {write_speed:.1f} MB/s", (read_speed, write_speed)
        except Exception as e:
            return f"Test failed: {str(e)}", (0, 0)

    def evaluate_hardware(self):
        """Evaluates hardware and generates recommendations with score"""
        # GPU analysis
        if self.info.get("has_cuda", False):
            if "rtx" in self.info["gpu_name"] and self.info["gpu_mem"] >= 6:
                self.score += 35
                self.recommendations.append("✅ RTX GPU with at least 6GB memory - Ideal for processing")
            elif self.info["gpu_mem"] >= 8:
                self.score += 30
                self.recommendations.append("✅ GPU with at least 8GB memory - Excellent for processing")
            elif self.info["gpu_mem"] >= 4:
                self.score += 20
                self.recommendations.append("⚠️ GPU with 4-8GB memory - Adequate but you may encounter limitations with large images")
            else:
                self.warnings.append("⚠️ GPU with less than 4GB memory - You may encounter problems with large images")
                self.score += 10
        elif self.info.get("has_mps", False):
            # Apple Silicon MPS support
            self.score += 35
            self.recommendations.append("✅ Apple Metal Performance Shaders (MPS) - Excellent acceleration for Apple Silicon")
        else:
            # No accelerated GPU
            self.score += 5
            self.warnings.append("⚠️ No accelerated GPU detected - Processing will be much slower on CPU")

        # RAM analysis
        if self.info["ram_gb"] >= 16:
            self.score += 20
            self.recommendations.append("✅ 16GB or more RAM - Ideal for large image batches")
        elif self.info["ram_gb"] >= 8:
            self.score += 15
            self.recommendations.append("✅ 8GB or more RAM - Adequate for processing")
        elif self.info["ram_gb"] >= 6:
            self.score += 10
            self.recommendations.append("⚠️ 6-8GB RAM - Works but you may encounter limitations with large batches")
        else:
            self.warnings.append("⚠️ Less than 6GB RAM - You may encounter problems with image batches")
            self.score += 5

        # CPU analysis
        cpu_score = 0
        cpu_message = ""
        
        # Check for modern CPUs
        modern_cpu = any(brand in self.info["cpu_brand"] for brand in ["intel", "amd"]) and \
                     any(gen in self.info["cpu_brand"] for gen in ["i5", "i7", "i9", "ryzen 5", "ryzen 7", "ryzen 9"])
        
        # Check for Apple Silicon
        apple_silicon = "apple" in self.info["cpu_brand"] or "m1" in self.info["cpu_brand"] or "m2" in self.info["cpu_brand"]
        
        if apple_silicon:
            cpu_score = 25
            cpu_message = "✅ Apple Silicon processor (M1/M2) - Excellent for processing"
        elif modern_cpu:
            if self.info["cpu_cores"] >= 8:
                cpu_score = 20
                cpu_message = "✅ Modern processor with 8+ cores - Excellent for processing"
            elif self.info["cpu_cores"] >= 4:
                cpu_score = 15
                cpu_message = "✅ Modern processor with 4+ cores - Adequate for processing"
            else:
                cpu_score = 10
                cpu_message = "⚠️ Modern processor but with few cores - May be limited for large batches"
        else:
            if self.info["cpu_cores"] >= 4:
                cpu_score = 10
                cpu_message = "⚠️ Processor with 4+ cores"
            else:
                cpu_score = 5
                cpu_message = "⚠️ Processor with less than 4 cores - Limited performance"
                self.warnings.append("⚠️ CPU with less than 4 cores. Performance may be significantly limited.")
        
        self.score += cpu_score
        if cpu_message:
            self.recommendations.append(cpu_message)

        # Disk speed analysis
        if self.info["disk_speed_read"] >= 500 and self.info["disk_speed_write"] >= 300:
            self.score += 15
            self.recommendations.append("✅ Fast disk (NVMe SSD) - Excellent for loading/saving images")
        elif self.info["disk_speed_read"] >= 200 and self.info["disk_speed_write"] >= 100:
            self.score += 10
            self.recommendations.append("⚠️ Decent disk but could be improved with an NVMe SSD")
        else:
            self.score += 5
            self.recommendations.append("⚠️ Slow disk - An NVMe SSD would significantly improve performance")

        # Operating system check
        if self.info["os"] == "darwin":
            if apple_silicon:
                self.score += 5
                self.recommendations.append("✅ macOS system on Apple Silicon - Fully optimized support")
            else:
                self.score += 3
                self.recommendations.append("✅ macOS system - Good compatibility")

        # General recommendations
        self.recommendations.append("💡 If using a laptop, ensure it's well ventilated and connected to power")
        self.recommendations.append("💡 Consider using a cooling pad for laptops to avoid thermal throttling")

    def get_recommendation_level(self) -> Tuple[str, str]:
        """Returns recommendation level based on score"""
        percentage = (self.score / self.max_score) * 100
        
        if percentage >= 85:
            return "HIGHLY RECOMMENDED", "🟢"
        elif percentage >= 65:
            return "RECOMMENDED", "🟠"
        elif percentage >= 40:
            return "LIMITED USE", "🟡"
        else:
            return "NOT RECOMMENDED", "🔴"

    def generate_report(self) -> str:
        """Generates a complete report in markdown with score"""
        report = "## 🖥️ Hardware Specifications\n"
        report += f"- **GPU:** {self.info['gpu']}\n"
        report += f"- **CPU:** {self.info['cpu']}\n"
        report += f"- **RAM:** {self.info['ram']}\n"
        report += f"- **Disk speed:** {self.info['disk_speed']}\n\n"
        
        # Show score
        percentage = (self.score / self.max_score) * 100
        report += f"## 📊 Hardware Score: {self.score}/{self.max_score} ({percentage:.1f}%)\n\n"

        if self.warnings:
            report += "## ⚠️ Warnings\n"
            report += "\n".join(f"- {w}" for w in self.warnings) + "\n\n"

        if self.recommendations:
            report += "## 💡 Recommendations\n"
            report += "\n".join(f"- {r}" for r in self.recommendations) + "\n\n"

        # Overall evaluation
        level, emoji = self.get_recommendation_level()
        report = f"## {emoji} Recommendation Level: {level}\n" + report
        
        if level == "HIGHLY RECOMMENDED":
            report += "## ✅ Conclusion\n"
            report += "Your hardware is excellent for using PyPotteryInk! 🎉\n"
        elif level == "RECOMMENDED":
            report += "## ✅ Conclusion\n"
            report += "Your hardware is adequate for using PyPotteryInk, but you may encounter some limitations with very large images.\n"
        elif level == "LIMITED USE":
            report += "## ⚠️ Conclusion\n"
            report += "Your hardware can run PyPotteryInk but you may encounter significant performance limitations.\n"
        else:
            report += "## ❌ Conclusion\n"
            report += "Your hardware is not recommended for PyPotteryInk. Processing may be very slow or impossible.\n"
        
        report += "\n💡 *If you're using a laptop, ensure it's well ventilated and connected to power during processing*"

        return report

def run_hardware_check() -> str:
    """
    Public function to run hardware check
    Returns a formatted markdown report
    """
    checker = HardwareChecker()
    return checker.generate_report()
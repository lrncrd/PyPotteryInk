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
    """Classe interna per il check hardware"""
    def __init__(self):
        self.info = self.get_hardware_info()
        self.recommendations = []
        self.warnings = []
        self.evaluate_hardware()

    def get_hardware_info(self) -> Dict[str, str]:
        """Raccoglie informazioni sull'hardware del sistema"""
        info = {}

        # Informazioni GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["gpu"] = f"{gpu_name} ({gpu_mem:.1f} GB)"
            info["gpu_mem"] = gpu_mem
        else:
            info["gpu"] = "Nessuna GPU NVIDIA rilevata"
            info["gpu_mem"] = 0

        # Informazioni CPU
        if CPUINFO_AVAILABLE:
            cpu_info = cpuinfo.get_cpu_info()
            info["cpu"] = f"{cpu_info['brand_raw']} ({psutil.cpu_count()} cores)"
        else:
            info["cpu"] = f"{platform.processor()} ({psutil.cpu_count()} cores)"
        info["cpu_cores"] = psutil.cpu_count()

        # Informazioni RAM
        ram = psutil.virtual_memory().total / (1024**3)
        info["ram"] = f"{ram:.1f} GB"
        info["ram_gb"] = ram

        # Test velocità disco
        disk_speed, disk_speed_values = self.test_disk_speed()
        info["disk_speed"] = disk_speed
        info["disk_speed_read"] = disk_speed_values[0]
        info["disk_speed_write"] = disk_speed_values[1]

        return info

    def test_disk_speed(self) -> Tuple[str, Tuple[float, float]]:
        """Testa la velocità di lettura/scrittura del disco"""
        try:
            test_file = "speed_test.tmp"
            start_time = time.time()

            # Scrittura
            with open(test_file, "wb") as f:
                f.write(os.urandom(100 * 1024 * 1024))  # 100MB
            write_time = time.time() - start_time

            # Lettura
            start_time = time.time()
            with open(test_file, "rb") as f:
                _ = f.read()
            read_time = time.time() - start_time

            os.remove(test_file)

            write_speed = (100 / write_time)  # MB/s
            read_speed = (100 / read_time)    # MB/s
            return f"Lettura: {read_speed:.1f} MB/s | Scrittura: {write_speed:.1f} MB/s", (read_speed, write_speed)
        except Exception as e:
            return f"Test fallito: {str(e)}", (0, 0)

    def evaluate_hardware(self):
        """Valuta l'hardware e genera raccomandazioni"""
        # Analisi GPU
        if torch.cuda.is_available():
            if self.info["gpu_mem"] < 4:
                self.warnings.append("⚠️ La tua GPU ha meno di 4GB di memoria. Potresti incontrare problemi con immagini grandi.")
            elif self.info["gpu_mem"] < 8:
                self.recommendations.append("ℹ️ Per prestazioni ottimali con immagini ad alta risoluzione, considera una GPU con almeno 8GB.")
        else:
            self.warnings.append("⚠️ Nessuna GPU NVIDIA rilevata. L'elaborazione sarà molto più lenta sulla CPU.")

        # Analisi RAM
        if self.info["ram_gb"] < 8:
            self.warnings.append("⚠️ Hai meno di 8GB di RAM. Potresti incontrare problemi con batch di immagini grandi.")
        elif self.info["ram_gb"] < 16:
            self.recommendations.append("ℹ️ Per lavorare con batch di immagini grandi, 16GB o più di RAM sono consigliati.")

        # Analisi CPU
        if self.info["cpu_cores"] < 4:
            self.warnings.append("⚠️ CPU con meno di 4 core. Le prestazioni potrebbero essere limitate.")

        # Analisi velocità disco
        if self.info["disk_speed_read"] < 200 or self.info["disk_speed_write"] < 100:
            self.recommendations.append("ℹ️ Un disco più veloce (SSD NVMe) migliorerebbe le prestazioni con grandi quantità di immagini.")

        # Raccomandazioni generali
        self.recommendations.append("💡 Per lavorare con un portatile, assicurati di collegarlo alla corrente e di usare una superficie ben ventilata.")
        self.recommendations.append("💡 Se usi un portatile, considera l'uso di una base di raffreddamento per evitare thermal throttling.")

    def generate_report(self) -> str:
        """Genera un report completo in markdown"""
        report = "## 🖥️ Specifiche Hardware\n"
        report += f"- **GPU:** {self.info['gpu']}\n"
        report += f"- **CPU:** {self.info['cpu']}\n"
        report += f"- **RAM:** {self.info['ram']}\n"
        report += f"- **Velocità disco:** {self.info['disk_speed']}\n\n"

        if self.warnings:
            report += "## ⚠️ Avvertenze\n"
            report += "\n".join(f"- {w}" for w in self.warnings) + "\n\n"

        if self.recommendations:
            report += "## 💡 Raccomandazioni\n"
            report += "\n".join(f"- {r}" for r in self.recommendations)

        # Valutazione complessiva
        if not torch.cuda.is_available() or self.info["ram_gb"] < 8 or (torch.cuda.is_available() and self.info["gpu_mem"] < 4):
            report = "## ❌ Hardware non raccomandato\n" + report
            report += "\n\n🔴 ATTENZIONE: Il tuo hardware potrebbe non essere sufficiente per un utilizzo ottimale."
        else:
            report = "## ✅ Hardware adeguato\n" + report
            report += "\n\n🟢 Il tuo hardware soddisfa i requisiti minimi per l'utilizzo."

        return report

def run_hardware_check() -> str:
    """
    Funzione pubblica per eseguire il check hardware
    Restituisce un report formattato in markdown
    """
    checker = HardwareChecker()
    return checker.generate_report()

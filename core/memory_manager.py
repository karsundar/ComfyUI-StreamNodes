"""Non-invasive system and accelerator memory reporting."""

import torch


def memory_report():
    report = {
        "cuda_available": torch.cuda.is_available(),
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "device_name": "CPU",
    }

    if torch.cuda.is_available():
        try:
            report["allocated_bytes"] = torch.cuda.memory_allocated()
            report["reserved_bytes"] = torch.cuda.memory_reserved()
            report["device_name"] = torch.cuda.get_device_name()
        except Exception:
            pass

    return report

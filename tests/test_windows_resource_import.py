"""Windows compatibility regression: V8 imports without Unix-only resource."""

import pytest
import torch
import mbmf_q1_infrastructure_v8 as pub


def test_v8_import_without_unix_resource_module():
    """Regression for the Windows ModuleNotFoundError at import time.

    Preload psutil first because psutil itself uses resource on Linux; Windows
    psutil does not. Then block resource only while importing V8 and measuring
    RSS to simulate the Windows import constraint.
    """
    import subprocess
    import sys

    probe = r'''
import builtins
import torch
import psutil
_real_import = builtins.__import__
def without_unix_resource(name, *args, **kwargs):
    if name == "resource":
        raise ModuleNotFoundError("No module named 'resource' (Windows simulation)")
    return _real_import(name, *args, **kwargs)
builtins.__import__ = without_unix_resource
import mbmf_q1_infrastructure_v8 as pub
assert pub.PeakMemoryMonitor(torch.device("cpu"))._rss() > 0
'''
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(pub.THIS_DIR),
        capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_windows_missing_psutil_memory_monitor_message(monkeypatch):
    """Windows fallback is explicit and does not try to import Unix resource."""
    import builtins
    import sys

    real_import = builtins.__import__
    def without_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("simulated missing psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_psutil)
    monkeypatch.setattr(pub.os, "name", "nt")
    with pytest.raises(RuntimeError, match="psutil"):
        pub.PeakMemoryMonitor(torch.device("cpu"))._rss()

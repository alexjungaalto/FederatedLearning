#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:03:47 2025

@author: junga1
"""

import subprocess
import time

# Define worker topology (neighbor connections)
topology = {
    1: {"a": 1.0, "neighbors": [2, 3]},
    2: {"a": 2.0, "neighbors": [1, 3]},
    3: {"a": 3.0, "neighbors": [1, 2, 4]},
    4: {"a": 4.0, "neighbors": [3]}
}

# Launch workers
processes = []
for worker_id, data in topology.items():
    cmd = ["python", "WorkerGDAsync.py", str(worker_id), str(data["a"])] + list(map(str, data["neighbors"]))
    p = subprocess.Popen(cmd)
    processes.append(p)
    time.sleep(1)  # Staggered start

# Wait for all processes
for p in processes:
    p.wait()

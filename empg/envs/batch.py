"""Ordered environment batches; ALFWorld workers own their own native planner.

Workers are lightweight Python subprocesses, so they neither import torch nor
inherit a CUDA context. Only observations/info cross the JSON pipe, never model
weights. Each worker handles a fixed subset of the episode slots.
"""

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor


class SerialBatch:
    def __init__(self, bench, tasks):
        self.bench, self.tasks, self.envs = bench, tasks, []

    def reset(self):
        for task in self.tasks:
            self.envs.append(self.bench.make_env(task))
        return [env.reset() for env in self.envs]

    def step(self, active, actions):
        return [self.envs[i].step(a) for i, a in zip(active, actions)]

    def close(self):
        for env in self.envs:
            env.close()
        self.envs.clear()


class _Worker:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "empg.envs.worker"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, bufsize=1)

    def call(self, message):
        with self.lock:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"ALFWorld worker exited ({self.process.poll()})")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"ALFWorld worker: {result['error']}")
        return result["result"]

    def close(self):
        try:
            if self.process.poll() is None:
                self.call({"cmd": "exit"})
                self.process.wait(timeout=10)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            self.process.stdin.close()
            self.process.stdout.close()


class ALFWorldPool:
    def __init__(self, workers):
        self.workers = [_Worker() for _ in range(workers)]
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def calls(self, messages):
        futures = [self.executor.submit(w.call, m) if m is not None else None
                   for w, m in zip(self.workers, messages)]
        return [f.result() if f is not None else [] for f in futures]

    def close(self):
        self.executor.shutdown(wait=True)
        for worker in self.workers:
            worker.close()


class ParallelBatch:
    def __init__(self, pool, tasks):
        self.pool = pool
        self.slots = [[] for _ in pool.workers]
        for i, task in enumerate(tasks):
            self.slots[i % len(self.slots)].append((i, task))

    def reset(self):
        results = self.pool.calls([
            {"cmd": "reset", "tasks": [t for _, t in slots]} for slots in self.slots])
        ordered = [None] * sum(map(len, self.slots))
        for slots, rows in zip(self.slots, results):
            for (i, _), row in zip(slots, rows):
                ordered[i] = row
        return ordered

    def step(self, active, actions):
        requested = dict(zip(active, actions))
        local = [[(j, requested[i]) for j, (i, _) in enumerate(slots) if i in requested]
                 for slots in self.slots]
        results = self.pool.calls([{"cmd": "step", "actions": rows} if rows else None
                                   for rows in local])
        ordered = {}
        for slots, indices, rows in zip(self.slots, local, results):
            for (j, _), row in zip(indices, rows):
                ordered[slots[j][0]] = row
        return [ordered[i] for i in active]

    def close(self):
        self.pool.calls([{"cmd": "close"} for _ in self.slots])

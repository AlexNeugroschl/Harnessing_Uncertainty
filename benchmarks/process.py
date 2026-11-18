"""Run a phase, stopping its Linux process tree on timeout, including nested sessions."""

import os
import signal
import subprocess
from pathlib import Path


def _linux_tree(root):
    """Snapshot descendant identities before a terminated parent can orphan them."""
    processes = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(") ", 1)[1].split()
            processes[int(path.parent.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root}
    while True:
        found = {pid for pid, (parent, _) in processes.items() if parent in descendants}
        if found <= descendants:
            break
        descendants.update(found)
    return {pid: processes[pid][1] for pid in descendants if pid in processes}


def _signal_snapshot(snapshot, sig):
    for pid, started in snapshot.items():
        try:
            # Guard against PID reuse after the initial process exited.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            if fields[19] == started:
                os.kill(pid, sig)
        except (OSError, ValueError, IndexError):
            pass


def run_process(cmd, timeout=None):
    if timeout is not None and timeout <= 0:
        raise subprocess.TimeoutExpired(cmd, timeout)
    process = subprocess.Popen(cmd, start_new_session=os.name == "posix")
    try:
        code = process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        descendants = _linux_tree(process.pid) if Path("/proc/self/stat").exists() else {}
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            _signal_snapshot(descendants, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # A phase can exit before its children. Kill the entire group even
            # when the immediate parent already handled SIGTERM and exited.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # Nested runners create their own sessions. Killing only the
                # outer process group would leave their GPU workers running.
                _signal_snapshot(descendants, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
            process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, cmd)

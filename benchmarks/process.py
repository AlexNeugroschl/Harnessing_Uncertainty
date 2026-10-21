"""Run a phase, stopping its Linux process group (including env workers) on timeout."""

import os
import signal
import subprocess


def run_process(cmd, timeout=None):
    if timeout is not None and timeout <= 0:
        raise subprocess.TimeoutExpired(cmd, timeout)
    process = subprocess.Popen(cmd, start_new_session=os.name == "posix")
    try:
        code = process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
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
            elif process.poll() is None:
                process.kill()
            process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, cmd)

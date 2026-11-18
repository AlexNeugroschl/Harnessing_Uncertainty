import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from benchmarks.process import run_process


def test_nonzero_exit_preserves_failure():
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_process([sys.executable, "-c", "raise SystemExit(7)"], timeout=5)
    assert error.value.returncode == 7


@pytest.mark.skipif(os.name != "posix", reason="ALFWorld runs in Linux/WSL")
def test_timeout_stops_descendants(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = ("import subprocess,sys,time; "
            "from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)")
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", code, str(pid_file)], timeout=1)
    assert time.monotonic() - start < 8
    pid = int(pid_file.read_text())
    # An orphan may await init reaping briefly; a zombie cannot keep working.
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="GPU phases run in Linux/WSL")
def test_outer_deadline_stops_nested_runner_with_separate_session(tmp_path):
    pid_file = tmp_path / "nested.pid"
    child = ("import os,sys,time; from pathlib import Path; "
             "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)")
    nested = ("import sys; from benchmarks.process import run_process; "
              f"run_process([sys.executable, '-c', {child!r}, sys.argv[1]], timeout=60)")
    with pytest.raises(subprocess.TimeoutExpired):
        run_process([sys.executable, "-c", nested, str(pid_file)], timeout=2)
    pid = int(pid_file.read_text())
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z")

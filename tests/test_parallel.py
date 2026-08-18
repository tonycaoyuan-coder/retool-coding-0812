from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest

from retool_coding_0812.parallel import run_parallel


class ParallelTests(unittest.TestCase):
    def test_jobs_overlap(self) -> None:
        started = time.perf_counter()
        run_parallel(
            [
                [sys.executable, "-c", "import time; time.sleep(0.4)"],
                [sys.executable, "-c", "import time; time.sleep(0.4)"],
                [sys.executable, "-c", "import time; time.sleep(0.4)"],
            ],
            cwd=Path.cwd(),
            labels=("c0", "c1", "c2"),
        )
        self.assertLess(time.perf_counter() - started, 1.0)

    def test_failure_terminates_sibling_process_group(self) -> None:
        with TemporaryDirectory() as raw:
            marker = Path(raw) / "orphan.txt"
            sleeper = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable,'-c',\"import pathlib,time; time.sleep(2); "
                f"pathlib.Path({str(marker)!r}).write_text('orphan')\"]); time.sleep(10)"
            )
            with self.assertRaises(subprocess.CalledProcessError):
                run_parallel(
                    [
                        [sys.executable, "-c", "import sys,time; time.sleep(0.2); sys.exit(7)"],
                        [sys.executable, "-c", sleeper],
                    ],
                    cwd=Path.cwd(),
                    labels=("fail", "sibling"),
                )
            time.sleep(2.2)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()

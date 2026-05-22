from __future__ import annotations

import subprocess
import sys


def main() -> None:
    args = sys.argv[1:]
    command = [sys.executable, "scripts/run_hybridsn_small_fewshot.py"] + args
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

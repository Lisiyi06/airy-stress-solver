"""Run the text interface directly without configuring PYTHONPATH."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from airy_solver.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
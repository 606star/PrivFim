"""Experiment 3: clients. All options are forwarded to the unified runner."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.cli import main

if __name__ == "__main__":
    main(["suite", "--experiments", "3", *sys.argv[1:]])

import sys
from pathlib import Path

# The node package is a plain ComfyUI custom_nodes directory, not an installed distribution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

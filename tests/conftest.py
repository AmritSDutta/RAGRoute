import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project root (one level above tests/)
sys.path.insert(0, str(ROOT))                # <- add project root, NOT the src folder

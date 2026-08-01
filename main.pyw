import os, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

from Orbital_Knight.gui import main


if __name__ == "__main__":
    main()

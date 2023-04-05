import sys
from pathlib import Path

TRAJECTRON_PATHLIB = Path("../Trajectron-plus-plus/trajectron").resolve()
assert TRAJECTRON_PATHLIB.is_dir(), "Could not import Trajectron++"
sys.path.append(str(TRAJECTRON_PATHLIB))

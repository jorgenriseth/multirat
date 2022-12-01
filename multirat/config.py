from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MESH_DIR = ROOT_DIR / "mesh"
RESULTS_DIR = ROOT_DIR / "results"

# TODO: Use a proper python settings class here.
# TODO: Currently it assumes the location where the package is installed.
# 	Should allow package installation elsewhere. Currently using
# relative paths in notebook to circumvent.

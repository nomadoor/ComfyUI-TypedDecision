import importlib.util
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK.parents[1]))  # ComfyUI root, for comfy / comfy_api / folder_paths

spec = importlib.util.spec_from_file_location("typed_decision", PACK / "__init__.py", submodule_search_locations=[str(PACK)])
module = importlib.util.module_from_spec(spec)
sys.modules["typed_decision"] = module
spec.loader.exec_module(module)

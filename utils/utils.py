from pathlib import Path
import yaml
from typing import Any, Dict, List, TypedDict
from config.settings import MODELS_PATH

def load_yaml() -> Dict[str, Any]:
    """Is for loading yaml files, but we use it just for models"""
    p = Path(MODELS_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Does not exist: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

class ModelSpec(TypedDict, total=False):
    name: str
    url: str

class LoadModel:
    def __init__(self):
        self.data = load_yaml()

    def extract_section_models(self, section: str) -> List[ModelSpec]:
        "We take the values from the yaml file"
        items = self.data.get(section, [])
        if not isinstance(items, list):
            raise ValueError(f"The section '{section}' is not a list")
        
        out: List[ModelSpec] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            out.append({
                "name": item.get("name", ""),
                "url": item.get("url", "")
            })
        return out

    def ensure_model(self, section: str) -> List[Path]:
        """ Ensure the model directory exists, return a List of paths or an error message """
        # Generate a variable to save the project path
        base_dir = Path.home() / ".cache" / "Local-LLM-for-Robots"
        models = []
        values = self.extract_section_models(section)
        for value in values:     
            model_dir = base_dir / section / value.get('name')
            if not model_dir.exists():
                raise FileNotFoundError(f"[LLM_LOADER] Direct path does not exist: {model_dir}\n")
            models.append(Path(model_dir))
        return models

 #———— Example Usage ————
if "__main__" == __name__:
    loader = LoadModel()
    model = loader.ensure_model("stt")
    print(model[0])
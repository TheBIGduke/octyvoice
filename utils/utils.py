from pathlib import Path
import yaml
from typing import Any, Dict, List, TypedDict
from config.settings import MODELS_PATH


def load_yaml() -> Dict[str, Any]:
    """Load YAML configuration file for models."""
    p = Path(MODELS_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {p}")
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
        """Extract model specifications from a section in the YAML file."""
        items = self.data.get(section, [])
        if not isinstance(items, list):
            raise ValueError(f"Section '{section}' is not a list in configuration")
        
        out: List[ModelSpec] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append({
                "name": item.get("name", ""),
                "url": item.get("url", "")
            })
        return out

    def ensure_model(self, section: str) -> List[Path]:
        """
        Ensure the model files exist in the cache directory.
        Returns a list of paths to the model files.
        """
        base_dir = Path.home() / ".cache" / "Local-LLM-for-Robots"
        models = []
        values = self.extract_section_models(section)
        
        for value in values:     
            model_dir = base_dir / section / value.get('name')
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Model file does not exist: {model_dir}\n"
                    f"Run 'bash utils/download_models.sh' to download models."
                )
            models.append(model_dir)
        return models


# Example Usage
if __name__ == "__main__":
    loader = LoadModel()
    model = loader.ensure_model("stt")
    print(f"STT model path: {model[0]}")
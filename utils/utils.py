from pathlib import Path
import yaml
import os
from typing import Any, Dict, List, TypedDict, Union
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

    def ensure_model(self, section: str, model_name: str | None = None) -> Union[List[Path], Path]:
        """
        Ensure the model files exist in the cache directory.
        If model_name is provided, returns the Path to that specific model.
        Otherwise, returns a list of all model Paths in the section.
        """
        # Check for environment variable override, otherwise use default path
        cache_env = os.getenv('OCTYVOICE_CACHE')
        if cache_env:
            base_dir = Path(cache_env)
        else:
            base_dir = Path.home() / ".cache" / "OctyVoice"

        models = []
        values = self.extract_section_models(section)
        
        for value in values:     
            name = value.get('name')
            if not name:
                continue

            model_dir = base_dir / section / name

            if model_name and name == model_name:
                # Specific model lookup
                if not model_dir.exists():
                    raise FileNotFoundError(
                        f"Specific model file '{model_name}' does not exist at {model_dir}\n"
                        f"Run 'bash utils/download_models.sh' to download models."
                    )
                return model_dir
            elif not model_name:
                # List all models
                if not model_dir.exists():
                    raise FileNotFoundError(
                        f"Model file does not exist: {model_dir}\n"
                        f"Run 'bash utils/download_models.sh' to download models."
                    )
                models.append(model_dir)

        if model_name and not models:
            raise FileNotFoundError(f"Model '{model_name}' not found in config or cache.")
            
        return models


# Example Usage
if __name__ == "__main__":
    loader = LoadModel()
    model = loader.ensure_model("stt")
    print(f"STT model path: {model[0]}")
    
    # Test name lookup
    small_whisper = loader.ensure_model("stt", "small.pt")
    print(f"Small Whisper path: {small_whisper}")
from pathlib import Path
import yaml
import os
import logging
from typing import Any, Dict, List, TypedDict, Union
from config.settings import MODELS_PATH


def load_yaml() -> Dict[str, Any]:
    """Load YAML configuration file for models."""
    p = Path(MODELS_PATH)
    
    if not p.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {p}")
    
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {p}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read configuration file {p}: {e}")
    
    return data or {}


class ModelSpec(TypedDict, total=False):
    name: str
    url: str


class LoadModel:
    def __init__(self):
        self.log = logging.getLogger("LoadModel")
        
        try:
            self.data = load_yaml()
        except Exception as e:
            self.log.error(f"Failed to load model configuration: {e}")
            raise
        
        # Check for environment variable override
        cache_env = os.getenv('OCTYVOICE_CACHE')
        if cache_env:
            self.base_dir = Path(cache_env)
            self.log.info(f"Using custom cache directory: {self.base_dir}")
        else:
            self.base_dir = Path.home() / ".cache" / "OctyVoice"
            self.log.debug(f"Using default cache directory: {self.base_dir}")

    def extract_section_models(self, section: str) -> List[ModelSpec]:
        """Extract model specifications from a section in the YAML file."""
        if section not in self.data:
            raise ValueError(f"Section '{section}' not found in configuration. Available sections: {list(self.data.keys())}")
        
        items = self.data.get(section, [])
        
        if not isinstance(items, list):
            raise ValueError(f"Section '{section}' is not a list in configuration")
        
        out: List[ModelSpec] = []
        for item in items:
            if not isinstance(item, dict):
                self.log.warning(f"Skipping non-dict item in section '{section}'")
                continue
            
            name = item.get("name", "")
            url = item.get("url", "")
            
            if not name:
                self.log.warning(f"Skipping model with no name in section '{section}'")
                continue
            
            out.append({
                "name": name,
                "url": url
            })
        
        return out

    def ensure_model(self, section: str, model_name: str | None = None) -> Union[List[Path], Path]:
        """
        Ensure the model files exist in the cache directory.
        If model_name is provided, returns the Path to that specific model.
        Otherwise, returns a list of all model Paths in the section.
        
        Args:
            section: Section name in models.yml (e.g., "stt", "tts")
            model_name: Optional specific model filename to lookup
            
        Returns:
            Path to specific model, or list of Paths to all models in section
            
        Raises:
            FileNotFoundError: If model(s) don't exist in cache
            ValueError: If section is invalid
        """
        try:
            values = self.extract_section_models(section)
        except ValueError as e:
            self.log.error(str(e))
            raise
        
        if not values:
            raise ValueError(f"No models found in section '{section}'")
        
        models = []
        
        for value in values:     
            name = value.get('name')
            if not name:
                continue

            model_dir = self.base_dir / section / name

            if model_name and name == model_name:
                # Specific model lookup
                if not model_dir.exists():
                    raise FileNotFoundError(
                        f"Model file '{model_name}' does not exist at {model_dir}\n"
                        f"Run 'bash utils/download_models.sh' to download models."
                    )
                self.log.debug(f"Found model: {model_dir}")
                return model_dir
                
            elif not model_name:
                # List all models
                if not model_dir.exists():
                    raise FileNotFoundError(
                        f"Model file does not exist: {model_dir}\n"
                        f"Run 'bash utils/download_models.sh' to download models."
                    )
                models.append(model_dir)

        if model_name:
            # Specific model requested but not found
            available = [v['name'] for v in values if v.get('name')]
            raise FileNotFoundError(
                f"Model '{model_name}' not found in section '{section}'.\n"
                f"Available models: {available}"
            )
            
        return models

    def list_available_models(self, section: str) -> List[str]:
        """List all model names in a section."""
        try:
            values = self.extract_section_models(section)
            return [v['name'] for v in values if v.get('name')]
        except Exception as e:
            self.log.error(f"Failed to list models in section '{section}': {e}")
            return []

    def validate_all_models(self) -> Dict[str, List[str]]:
        """
        Check which models are missing from cache.
        
        Returns:
            Dict mapping section names to lists of missing model names
        """
        missing = {}
        
        for section in self.data.keys():
            try:
                values = self.extract_section_models(section)
                missing_in_section = []
                
                for value in values:
                    name = value.get('name')
                    if not name:
                        continue
                    
                    model_path = self.base_dir / section / name
                    if not model_path.exists():
                        missing_in_section.append(name)
                
                if missing_in_section:
                    missing[section] = missing_in_section
                    
            except Exception as e:
                self.log.warning(f"Error validating section '{section}': {e}")
        
        return missing


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    try:
        loader = LoadModel()
        
        # List all models
        print("\n=== Available Models ===")
        for section in ["stt", "tts"]:
            models = loader.list_available_models(section)
            print(f"{section.upper()}: {models}")
        
        # Validate all models
        print("\n=== Validating Models ===")
        missing = loader.validate_all_models()
        if missing:
            print("Missing models:")
            for section, models in missing.items():
                print(f"  {section}: {models}")
            print("\nRun 'bash utils/download_models.sh' to download missing models.")
        else:
            print("All models present in cache!")
        
        # Test specific model lookup
        print("\n=== Testing Specific Model Lookup ===")
        stt_model = loader.ensure_model("stt", "small.pt")
        print(f"Small Whisper model: {stt_model}")
        
        # Test listing all in section
        print("\n=== Testing Section Listing ===")
        all_stt = loader.ensure_model("stt")
        print(f"All STT models: {all_stt}")
        
    except Exception as e:
        print(f"Error: {e}")
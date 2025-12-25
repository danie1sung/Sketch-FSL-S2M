import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse YAML file at {config_path}. Error: {e}")
        return {}

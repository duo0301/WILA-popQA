import os
from pathlib import Path
import logging

def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    
def get_input_files(base_dir, pattern=None):
    """Get all input files matching a pattern from the base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logging.warning(f"Base directory {base_dir} does not exist")
        return []
    
    if pattern:
        # Search recursively through model subdirectories
        return list(base_path.glob(f"**/{pattern}"))
    else:
        # Get all CSV files recursively
        return list(base_path.glob("**/*.csv"))
    
def get_output_path(input_file, output_dir, suffix):
    """Generate output path based on input file and desired suffix."""
    # Extract model name from the directory structure
    input_path = Path(input_file)
    
    # If the input is in Inputs/model_name/files structure
    if "Inputs" in input_path.parts:
        model_idx = input_path.parts.index("Inputs") + 1
        if model_idx < len(input_path.parts):
            model_name = input_path.parts[model_idx]
        else:
            model_name = "unknown_model"
    else:
        # Fallback to extracting from filename
        parts = input_path.name.split('_')
        if len(parts) >= 3:
            model_name = parts[2].split('.')[0]  # Get model name without extension
        else:
            model_name = "unknown_model"
    
    # Create model-specific output directory
    model_output_dir = Path(output_dir) / model_name / "outputs"
    ensure_directory_exists(model_output_dir)
    
    # Create output filename
    output_filename = input_path.stem + suffix + input_path.suffix
    return model_output_dir / output_filename 
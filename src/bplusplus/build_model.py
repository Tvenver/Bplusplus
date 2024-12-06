import os
import shutil
import tempfile
from typing import Any

from .collect import Group, collect
from .train_validate import train_validate


def build_model(group_by_key: Group, search_parameters: dict[str, Any], images_per_group: int, model_output_folder: str):
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Do more work...
        print(f"Temporary directory path: {temp_dir}")

        groups = search_parameters.get(group_by_key.value, list[str])

        collect_images (
            group_by_key=group_by_key, 
            search_parameters=search_parameters, 
            images_per_group=images_per_group, 
            output_directory=temp_dir
        )
        
        train_validate(
            groups=groups, 
            dataset_path=temp_dir, 
            output_directory=model_output_folder
        )

    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil. rmtree (temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
    
    temp_dir = None
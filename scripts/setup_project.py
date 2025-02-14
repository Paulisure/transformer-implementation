import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure."""
    
    # Base directories
    directories = [
        'src/transformer/v1_basic_attention',
        'src/transformer/v2_multi_head',
        'src/transformer/v3_full_model',
        'tutorials/tutorial_1_basic_attention/outputs',
        'tutorials/tutorial_2_multi_head/outputs',
        'tutorials/tutorial_3_position_encoding/outputs',
        'tutorials/tutorial_4_full_transformer/outputs',
        'examples/attention_visualization',
        '.vscode'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Potential old locations to check (relative to project root)
    old_locations = {
        'visualize.py': 'examples/attention_visualization/visualize.py',
        'attention.py': 'src/transformer/v1_basic_attention/attention.py',
        'test_attention.py': 'src/transformer/v1_basic_attention/test_attention.py',
    }
    
    # Check and move files from old locations if they exist
    project_root = Path.cwd()
    for old_name, new_path in old_locations.items():
        # Look for the file in the project root or its immediate subdirectories
        possible_old_locations = [
            project_root / old_name,
            *project_root.glob(f"*/{old_name}")
        ]
        
        old_file = None
        for loc in possible_old_locations:
            if loc.exists() and str(loc) != str(project_root / new_path):
                old_file = loc
                break
        
        if old_file:
            new_file = project_root / new_path
            if not new_file.exists():
                new_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_file, new_file)
                print(f"Moved file: {old_file} -> {new_file}")
                # Optionally remove the old file
                # old_file.unlink()
            else:
                print(f"File already exists at destination: {new_file}")
        else:
            print(f"No existing file found for: {old_name}")
    
    # Create necessary __init__.py files
    init_locations = [
        'src/transformer',
        'src/transformer/v1_basic_attention',
        'src/transformer/v2_multi_head',
        'src/transformer/v3_full_model'
    ]
    
    for location in init_locations:
        init_file = Path(location) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"Created __init__.py in {location}")
        else:
            print(f"__init__.py already exists in {location}")

if __name__ == "__main__":
    create_directory_structure()
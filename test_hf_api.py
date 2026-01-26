"""
Test script to verify HuggingFace API responses.
Run this to see what data is returned when searching and fetching model files.
"""

from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

def test_search(query: str = "comfy", limit: int = 10):
    """Test model search functionality."""
    print(f"\n{'='*60}")
    print(f"TESTING SEARCH: '{query}'")
    print(f"{'='*60}\n")

    api = HfApi()

    try:
        models = api.list_models(
            search=query,
            limit=limit,
            sort="downloads",
            direction=-1
        )
        models_list = list(models)

        print(f"Found {len(models_list)} models:\n")

        for i, model in enumerate(models_list, 1):
            print(f"{i}. {model.id}")
            print(f"   Downloads: {model.downloads:,}" if model.downloads else "   Downloads: N/A")
            print(f"   Tags: {', '.join(model.tags[:5]) if model.tags else 'N/A'}")
            print()

        return models_list

    except Exception as e:
        print(f"ERROR: {e}")
        return []


def test_get_files(model_id: str):
    """Test getting file list for a specific model."""
    print(f"\n{'='*60}")
    print(f"TESTING GET FILES: '{model_id}'")
    print(f"{'='*60}\n")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    try:
        # Get file list
        files = api.list_repo_files(model_id, repo_type="model")
        print(f"Total files: {len(files)}\n")

        # Group by directory
        root_files = []
        directories = {}

        for f in files:
            if "/" not in f:
                root_files.append(f)
            else:
                parts = f.split("/")
                top_dir = parts[0]
                if top_dir not in directories:
                    directories[top_dir] = []
                directories[top_dir].append(f)

        # Print root files
        if root_files:
            print("ROOT FILES:")
            for f in sorted(root_files):
                print(f"  - {f}")
            print()

        # Print directories
        if directories:
            print("DIRECTORIES:")
            for dir_name in sorted(directories.keys()):
                dir_files = directories[dir_name]
                print(f"\n  📁 {dir_name}/ ({len(dir_files)} files)")

                # Show subdirectories
                subdirs = set()
                for f in dir_files:
                    parts = f.split("/")
                    if len(parts) > 2:
                        subdirs.add(parts[1])

                if subdirs:
                    print(f"     Subdirectories: {', '.join(sorted(subdirs))}")

                # Show first few files
                print("     Sample files:")
                for f in sorted(dir_files)[:5]:
                    print(f"       - {f}")
                if len(dir_files) > 5:
                    print(f"       ... and {len(dir_files) - 5} more files")

        return files

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_get_file_sizes(model_id: str):
    """Test getting file sizes for a model."""
    print(f"\n{'='*60}")
    print(f"TESTING GET FILE SIZES: '{model_id}'")
    print(f"{'='*60}\n")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    try:
        repo_info = api.repo_info(model_id, repo_type="model", files_metadata=True)

        if repo_info.siblings:
            print(f"Files with metadata: {len(repo_info.siblings)}\n")

            total_size = 0
            for file_info in repo_info.siblings[:10]:
                size = file_info.size or 0
                total_size += size
                size_str = format_size(size)
                print(f"  {file_info.rfilename}: {size_str}")

            if len(repo_info.siblings) > 10:
                print(f"  ... and {len(repo_info.siblings) - 10} more files")

            # Calculate total
            total_size = sum(f.size or 0 for f in repo_info.siblings)
            print(f"\nTotal size: {format_size(total_size)}")
        else:
            print("No file metadata available")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes is None or size_bytes == 0:
        return "Unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


if __name__ == "__main__":
    # Test 1: Search for "comfy"
    models = test_search("comfy", limit=5)

    # Test 2: Get files for specific model
    test_model = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
    test_get_files(test_model)

    # Test 3: Get file sizes
    test_get_file_sizes(test_model)

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)

import os
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import GatedRepoError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="HuggingFace Model Downloader",
    page_icon="🤗",
    layout="wide"
)

# Initialize session state
if "selected_files" not in st.session_state:
    st.session_state.selected_files = set()
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "active_model" not in st.session_state:
    st.session_state.active_model = None
if "model_files" not in st.session_state:
    st.session_state.model_files = []
if "model_file_sizes" not in st.session_state:
    st.session_state.model_file_sizes = {}


def get_hf_token():
    """Get HuggingFace token from environment or session state."""
    if "hf_token_input" in st.session_state and st.session_state.hf_token_input:
        return st.session_state.hf_token_input
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def get_model_files(model_id: str, token: str = None):
    """Get list of files in a model repository."""
    try:
        api = HfApi(token=token)
        files = api.list_repo_files(model_id, repo_type="model")
        return list(files)
    except Exception as e:
        st.error(f"Error fetching file list: {str(e)}")
        return []


def get_file_sizes(model_id: str, token: str = None):
    """Get file sizes for all files in the repository."""
    try:
        api = HfApi(token=token)
        repo_info = api.repo_info(model_id, repo_type="model", files_metadata=True)
        file_sizes = {}
        if repo_info.siblings:
            for file_info in repo_info.siblings:
                file_sizes[file_info.rfilename] = file_info.size
        return file_sizes
    except Exception as e:
        st.error(f"Error fetching file sizes: {str(e)}")
        return {}


def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes is None or size_bytes == 0:
        return "Unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def search_models(query: str, token: str = None, limit: int = 20):
    """Search for models on HuggingFace Hub."""
    api = HfApi(token=token)
    try:
        models = api.list_models(
            search=query,
            limit=limit,
            sort="downloads",
            direction=-1
        )
        return list(models)
    except Exception as e:
        st.error(f"Error searching models: {str(e)}")
        return []


def download_model_files(
    model_id: str,
    files: list = None,
    local_dir: str = None,
    token: str = None,
    progress_placeholder=None
):
    """Download model files from HuggingFace Hub."""
    try:
        if files and len(files) > 0:
            downloaded_files = []
            total_files = len(files)

            for idx, file in enumerate(files):
                if progress_placeholder:
                    progress_placeholder.progress(
                        (idx) / total_files,
                        text=f"Downloading {file}... ({idx + 1}/{total_files})"
                    )

                kwargs = {
                    "repo_id": model_id,
                    "filename": file,
                    "token": token,
                }

                if local_dir:
                    kwargs["local_dir"] = local_dir

                downloaded_path = hf_hub_download(**kwargs)
                downloaded_files.append(downloaded_path)

            if progress_placeholder:
                progress_placeholder.progress(1.0, text="Download complete!")

            return downloaded_files
        else:
            if progress_placeholder:
                progress_placeholder.progress(0.0, text="Downloading entire model...")

            kwargs = {
                "repo_id": model_id,
                "token": token,
            }

            if local_dir:
                kwargs["local_dir"] = local_dir

            downloaded_path = snapshot_download(**kwargs)

            if progress_placeholder:
                progress_placeholder.progress(1.0, text="Download complete!")

            return [downloaded_path]

    except GatedRepoError:
        st.error("This is a gated model. Please provide a valid HuggingFace token with access.")
        return None
    except Exception as e:
        st.error(f"Error downloading: {str(e)}")
        return None


def render_selection_controls(model_id: str, files: list, file_sizes: dict):
    """Render selection controls (Select All, Deselect All, metrics)."""
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Select All", key=f"sel_all_{model_id}", use_container_width=True):
            st.session_state.selected_files = set(files)
            st.rerun()
    with col2:
        if st.button("Deselect All", key=f"desel_all_{model_id}", use_container_width=True):
            st.session_state.selected_files = set()
            st.rerun()

    selected_size = sum(file_sizes.get(f, 0) or 0 for f in st.session_state.selected_files)
    total_size = sum(file_sizes.get(f, 0) or 0 for f in files)

    with col3:
        st.metric(
            "Selected",
            f"{len(st.session_state.selected_files)} / {len(files)} files",
            f"{format_size(selected_size)} / {format_size(total_size)}"
        )


def render_file_list(model_id: str, files: list, file_sizes: dict):
    """Render the file list with checkboxes (scrollable area)."""
    # Group files by directory
    root_files = []
    directories = {}

    for file_path in files:
        size = file_sizes.get(file_path, 0) or 0

        if "/" not in file_path:
            root_files.append({"path": file_path, "name": file_path, "size": size})
        else:
            parts = file_path.split("/")
            top_dir = parts[0]

            if top_dir not in directories:
                directories[top_dir] = {"files": [], "subdirs": set(), "total_size": 0}

            directories[top_dir]["files"].append({
                "path": file_path,
                "display_name": "/".join(parts[1:]),
                "size": size
            })
            directories[top_dir]["total_size"] += size

            if len(parts) > 2:
                directories[top_dir]["subdirs"].add(parts[1])

    # Render root files
    if root_files:
        st.markdown("**Root files:**")
        for file_info in sorted(root_files, key=lambda x: x["name"]):
            file_path = file_info["path"]
            is_selected = file_path in st.session_state.selected_files

            if st.checkbox(
                f"{file_info['name']} ({format_size(file_info['size'])})",
                value=is_selected,
                key=f"f_{model_id}_{file_path}"
            ):
                st.session_state.selected_files.add(file_path)
            else:
                st.session_state.selected_files.discard(file_path)

    # Render directories
    for dir_name in sorted(directories.keys()):
        dir_data = directories[dir_name]
        dir_file_paths = [f["path"] for f in dir_data["files"]]
        all_selected = all(f in st.session_state.selected_files for f in dir_file_paths)
        subdir_info = f", {len(dir_data['subdirs'])} subdirs" if dir_data['subdirs'] else ""

        # Directory checkbox
        dir_selected = st.checkbox(
            f"**{dir_name}/** ({len(dir_data['files'])} files{subdir_info}, {format_size(dir_data['total_size'])})",
            value=all_selected,
            key=f"d_{model_id}_{dir_name}"
        )

        if dir_selected and not all_selected:
            for f in dir_file_paths:
                st.session_state.selected_files.add(f)
            st.rerun()
        elif not dir_selected and all_selected:
            for f in dir_file_paths:
                st.session_state.selected_files.discard(f)
            st.rerun()

        # Group by subdirectory
        subdir_files = {}
        direct_files = []

        for file_info in dir_data["files"]:
            display = file_info["display_name"]
            if "/" in display:
                subdir = display.split("/")[0]
                if subdir not in subdir_files:
                    subdir_files[subdir] = []
                subdir_files[subdir].append(file_info)
            else:
                direct_files.append(file_info)

        # Render subdirectories as expanders
        for subdir_name in sorted(subdir_files.keys()):
            subdir_list = subdir_files[subdir_name]
            subdir_paths = [f["path"] for f in subdir_list]
            subdir_size = sum(f["size"] for f in subdir_list)
            subdir_all_selected = all(f in st.session_state.selected_files for f in subdir_paths)

            with st.expander(f"{subdir_name}/ ({len(subdir_list)} files, {format_size(subdir_size)})"):
                # Select all in subdir
                sub_selected = st.checkbox(
                    f"Select all in {subdir_name}/",
                    value=subdir_all_selected,
                    key=f"sd_{model_id}_{dir_name}_{subdir_name}"
                )

                if sub_selected and not subdir_all_selected:
                    for f in subdir_paths:
                        st.session_state.selected_files.add(f)
                    st.rerun()
                elif not sub_selected and subdir_all_selected:
                    for f in subdir_paths:
                        st.session_state.selected_files.discard(f)
                    st.rerun()

                # Files in subdirectory
                for file_info in sorted(subdir_list, key=lambda x: x["display_name"]):
                    file_path = file_info["path"]
                    relative = "/".join(file_info["display_name"].split("/")[1:])
                    is_selected = file_path in st.session_state.selected_files

                    if st.checkbox(
                        f"{relative} ({format_size(file_info['size'])})",
                        value=is_selected,
                        key=f"f_{model_id}_{file_path}"
                    ):
                        st.session_state.selected_files.add(file_path)
                    else:
                        st.session_state.selected_files.discard(file_path)

        # Direct files in directory
        if direct_files:
            with st.expander(f"Files in {dir_name}/ ({len(direct_files)} files)"):
                for file_info in sorted(direct_files, key=lambda x: x["display_name"]):
                    file_path = file_info["path"]
                    is_selected = file_path in st.session_state.selected_files

                    if st.checkbox(
                        f"{file_info['display_name']} ({format_size(file_info['size'])})",
                        value=is_selected,
                        key=f"f_{model_id}_{file_path}"
                    ):
                        st.session_state.selected_files.add(file_path)
                    else:
                        st.session_state.selected_files.discard(file_path)


def render_download_section(model_id: str, local_dir: str, token: str):
    """Render download buttons and handle downloads (always visible)."""
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    selected_files_list = list(st.session_state.selected_files)

    with col1:
        download_selected = st.button(
            f"Download Selected ({len(selected_files_list)} files)",
            type="primary",
            disabled=len(selected_files_list) == 0,
            use_container_width=True,
            key=f"dl_sel_{model_id}"
        )

    with col2:
        download_all = st.button(
            "Download Entire Model",
            use_container_width=True,
            key=f"dl_all_{model_id}"
        )

    progress_placeholder = st.empty()
    result_placeholder = st.empty()

    if download_selected and selected_files_list:
        download_dir = local_dir if local_dir else None
        result = download_model_files(
            model_id=model_id,
            files=selected_files_list,
            local_dir=download_dir,
            token=token,
            progress_placeholder=progress_placeholder
        )
        if result:
            result_placeholder.success(f"Downloaded {len(result)} files!")

    elif download_all:
        download_dir = local_dir if local_dir else None
        result = download_model_files(
            model_id=model_id,
            files=None,
            local_dir=download_dir,
            token=token,
            progress_placeholder=progress_placeholder
        )
        if result:
            result_placeholder.success("Downloaded entire model!")


# ============================================
# Modal Dialog for File Browser
# ============================================
@st.dialog("Model Files", width="large")
def show_model_dialog(model_id: str, local_dir: str):
    """Show modal dialog with file browser."""
    token = get_hf_token()

    # Load files if not already loaded for this model
    if st.session_state.active_model != model_id:
        st.session_state.active_model = model_id
        st.session_state.selected_files = set()
        with st.spinner("Loading files..."):
            st.session_state.model_files = get_model_files(model_id, token=token)
            st.session_state.model_file_sizes = get_file_sizes(model_id, token=token)

    files = st.session_state.model_files
    file_sizes = st.session_state.model_file_sizes

    if not files:
        st.error("No files found or error loading files.")
        return

    st.success(f"Found {len(files)} files")

    # Show directory summary
    dirs = set()
    for f in files:
        if "/" in f:
            dirs.add(f.split("/")[0])
    if dirs:
        st.info(f"Directories: {', '.join(sorted(dirs))}")

    # Selection controls (always visible)
    render_selection_controls(model_id, files, file_sizes)

    # Scrollable file list
    with st.container(height=400):
        render_file_list(model_id, files, file_sizes)

    # Download section (always visible)
    render_download_section(model_id, local_dir, token)


# ============================================
# Main UI
# ============================================
st.title("HuggingFace Model Downloader")

# Sidebar configuration
with st.sidebar:
    st.header("Search")

    # Search section
    search_query = st.text_input(
        "Model Search",
        placeholder="e.g., llama, stable-diffusion",
    )
    search_btn = st.button("Search", type="primary", use_container_width=True)

    st.markdown("---")
    st.header("Settings")

    # Token
    st.subheader("Authentication")
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        st.success("Token found in environment")
    st.text_input(
        "HuggingFace Token",
        type="password",
        key="hf_token_input",
        help="Required for private/gated models",
        placeholder="hf_..."
    )

    # Download directory
    st.subheader("Download Location")
    default_local_dir = os.getenv("HF_LOCAL_DIR", "")
    use_local_dir = st.toggle(
        "Save to local directory",
        value=bool(default_local_dir),
        help="Toggle ON to save to a local directory"
    )

    if use_local_dir:
        local_dir = st.text_input(
            "Local Directory",
            value=default_local_dir or "/data/localmodels",
            help="Set HF_LOCAL_DIR in .env for default"
        )
        st.info(f"Save to: {local_dir}")
    else:
        local_dir = ""
        st.info("Using HF cache directory")

# Perform search
if search_btn and search_query:
    token = get_hf_token()
    with st.spinner("Searching..."):
        results = search_models(search_query, token=token)
        st.session_state.search_results = results
        st.session_state.selected_files = set()
        st.session_state.active_model = None

# Main content area - single column with search results
if st.session_state.search_results:
    st.subheader(f"Search Results ({len(st.session_state.search_results)} models)")

    for model in st.session_state.search_results:
        downloads_str = f"{model.downloads:,}" if model.downloads else "N/A"
        tags_str = ", ".join(model.tags[:5]) if model.tags else ""

        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.markdown(f"**{model.id}**")

        with col2:
            st.caption(f"{downloads_str} downloads | {tags_str[:40]}")

        with col3:
            if st.button("View", key=f"view_{model.id}", use_container_width=True):
                show_model_dialog(model.id, local_dir)

        st.markdown("---")
else:
    st.info("Use the search box in the sidebar to find models")

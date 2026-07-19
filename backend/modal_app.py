"""Modal deployment definition for the Tree Fish Flask API.

Run from the repository root with: modal deploy backend/modal_app.py
"""

from pathlib import Path

import modal

BACKEND_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "Flask==3.1.3",
        "flask-cors==6.0.2",
        "chess==1.11.2",
        "numpy==2.4.4",
        "torch==2.11.0",
    )
    .add_local_dir(BACKEND_DIR, remote_path="/root/tree-fish-backend")
)

app = modal.App("tree-fish-api")
checkpoints = modal.Volume.from_name("tree-fish-checkpoints", create_if_missing=True)
config = modal.Secret.from_name("tree-fish-config")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/checkpoints": checkpoints},
    secrets=[config],
    timeout=600,
    max_containers=1,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)
@modal.wsgi_app()
def web():
    import sys

    sys.path.insert(0, "/root/tree-fish-backend")
    from main import create_app

    return create_app("/checkpoints", checkpoint_saved=checkpoints.commit)

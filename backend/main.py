"""HTTP API for playing against Tree Fish.

The API is intentionally stateless: every engine request includes a FEN.  This
makes it safe to run several browser games against an autoscaling deployment.
"""

# modal volume put --env main tree-fish-checkpoints checkpoints\checkpoint_iter7000.pt /checkpoint_iter7000.pt

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

import chess
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from engine_service import EngineService


DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def create_app(
    checkpoint_dir: str | Path | None = None,
    checkpoint_saved: Callable[[], None] | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CHECKPOINT_BYTES", 750 * 1024 * 1024))
    CORS(app, resources={r"/*": {"origins": os.getenv("CORS_ORIGINS", "*").split(",")}})
    uploads_enabled = os.getenv("ALLOW_CHECKPOINT_UPLOADS", "true").lower() in {"1", "true", "yes"}

    service = EngineService(checkpoint_dir or os.getenv("CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))
    app.extensions["engine_service"] = service

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(service.device)}

    @app.get("/checkpoints")
    def checkpoints():
        return {
            "checkpoints": service.list_checkpoints(),
            "uploads_enabled": uploads_enabled,
        }

    @app.post("/checkpoints")
    def upload_checkpoint():
        if not uploads_enabled:
            return {"error": "Checkpoint uploads are disabled on this deployment."}, 403
        uploaded = request.files.get("checkpoint")
        if uploaded is None or not uploaded.filename:
            return {"error": "Attach a .pt file in the 'checkpoint' field."}, 400

        filename = secure_filename(uploaded.filename)
        if not filename.lower().endswith((".pt", ".pth")):
            return {"error": "Checkpoint filenames must end in .pt or .pth."}, 400

        try:
            item = service.save_checkpoint(uploaded.stream, filename)
            if checkpoint_saved:
                checkpoint_saved()
            return {"checkpoint": item}, 201
        except (ValueError, RuntimeError) as exc:
            return {"error": str(exc)}, 400

    @app.post("/engine/move")
    def engine_move():
        payload = request.get_json(silent=True) or {}
        try:
            result = service.choose_move(
                fen=payload.get("fen", ""),
                checkpoint=payload.get("checkpoint", ""),
                sims=payload.get("sims", 800),
            )
            return jsonify(result)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return {"error": str(exc)}, 400

    # Compatibility route for older frontend builds. New clients use /engine/move.
    @app.put("/send_move")
    def legacy_engine_move():
        payload = request.get_json(silent=True) or {}
        if not payload.get("fen"):
            return {"error": "This API now requires the current 'fen'."}, 400
        try:
            return jsonify(service.choose_move(
                payload["fen"], payload.get("checkpoint", ""), payload.get("sims", 800)
            ))
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return {"error": str(exc)}, 400

    @app.put("/send_move/reset_board")
    def legacy_reset():
        fen = (request.get_json(silent=True) or {}).get("fen", "")
        try:
            chess.Board(fen)
        except ValueError:
            return {"error": "Invalid FEN."}, 400
        return {"status": "ok"}

    @app.errorhandler(413)
    def too_large(_error):
        return {"error": "Checkpoint exceeds the configured upload size limit."}, 413

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Tree Fish API")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app(args.checkpoint_dir).run(host=args.host, port=args.port, debug=args.debug)

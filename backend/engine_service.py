"""Checkpoint management and serializable MCTS analysis."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import BinaryIO

import chess
import torch

from board_encoder import swap_move_color
from endgame_search import EndgameMinimax
from mcts import MCTS, MCTSNode
from network import ChessNet


C_PUCT = 5.0
MAX_SIMS = int(os.getenv("MAX_SIMS", "5000"))
TOP_MOVES = 10
TOP_LINES = 3
LINE_DEPTH = 5


class EngineService:
    def __init__(self, checkpoint_dir: str | Path, max_cached_models: int = 2):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        requested = os.getenv("TORCH_DEVICE", "cuda")
        self.device = torch.device(requested if requested.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.max_cached_models = max_cached_models
        self._models: OrderedDict[str, tuple[ChessNet, dict]] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._search_lock = threading.Lock()
        self.endgame_engine = EndgameMinimax()

    def list_checkpoints(self) -> list[dict]:
        items = []
        for path in sorted(self.checkpoint_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in {".pt", ".pth"}:
                stat = path.stat()
                items.append({"name": path.name, "size_bytes": stat.st_size})
        return items

    def _checkpoint_path(self, name: str) -> Path:
        safe_name = Path(name).name
        if not safe_name or safe_name != name or Path(safe_name).suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("Choose a valid checkpoint name from the server list.")
        path = (self.checkpoint_dir / safe_name).resolve()
        if path.parent != self.checkpoint_dir or not path.is_file():
            raise FileNotFoundError(f"Checkpoint '{safe_name}' was not found.")
        return path

    @staticmethod
    def _read_checkpoint(path: Path) -> tuple[dict, int, int, object]:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"Could not safely read checkpoint: {exc}") from exc
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError("Checkpoint must contain a model state dictionary under 'model'.")
        blocks = int(checkpoint.get("num_res_blocks", 4))
        channels = int(checkpoint.get("channels", 128))
        return checkpoint, blocks, channels, checkpoint.get("iteration", "?")

    def _load_model(self, name: str) -> tuple[ChessNet, dict]:
        path = self._checkpoint_path(name)
        cache_key = f"{name}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
        with self._cache_lock:
            cached = self._models.get(cache_key)
            if cached:
                self._models.move_to_end(cache_key)
                return cached

            checkpoint, blocks, channels, iteration = self._read_checkpoint(path)
            model = ChessNet(num_res_blocks=blocks, channels=channels)
            try:
                model.load_state_dict(checkpoint["model"])
            except Exception as exc:
                raise ValueError(f"Checkpoint weights do not match its declared architecture: {exc}") from exc
            model = model.to(self.device).eval()
            metadata = {"iteration": iteration, "num_res_blocks": blocks, "channels": channels}
            self._models[cache_key] = (model, metadata)
            while len(self._models) > self.max_cached_models:
                self._models.popitem(last=False)
            return model, metadata

    def save_checkpoint(self, stream: BinaryIO, filename: str) -> dict:
        destination = self.checkpoint_dir / filename
        if destination.exists():
            raise ValueError(f"Checkpoint '{filename}' already exists; rename the file before uploading it.")
        temporary = self.checkpoint_dir / f".{filename}.{time.time_ns()}.upload"
        try:
            with temporary.open("wb") as output:
                while chunk := stream.read(8 * 1024 * 1024):
                    output.write(chunk)
            self._read_checkpoint(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        with self._cache_lock:
            for key in [key for key in self._models if key.startswith(f"{filename}:")]:
                del self._models[key]
        return {"name": filename, "size_bytes": destination.stat().st_size}

    @staticmethod
    def _actual_move(move: chess.Move, board: chess.Board) -> chess.Move:
        return swap_move_color(move) if board.turn == chess.BLACK else move

    def _line(self, root: MCTSNode, board: chess.Board, first_move: chess.Move) -> list[str]:
        line: list[str] = []
        node = root
        actual_board = board.copy(stack=False)
        canonical_move = first_move
        for _ in range(LINE_DEPTH):
            actual_move = self._actual_move(canonical_move, actual_board)
            if actual_move not in actual_board.legal_moves:
                break
            line.append(actual_board.san(actual_move))
            actual_board.push(actual_move)
            node = node.children.get(canonical_move)
            if node is None or not node.children:
                break
            canonical_move = max(node.children, key=lambda move: node.children[move].visit_count)
        return line

    def choose_move(self, fen: str, checkpoint: str, sims: int) -> dict:
        try:
            board = chess.Board(fen)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid FEN.") from exc
        if board.is_game_over():
            raise ValueError("The game is already over.")
        if not checkpoint:
            available = self.list_checkpoints()
            if not available:
                raise FileNotFoundError("No checkpoints are available. Upload one first.")
            checkpoint = available[-1]["name"]
        try:
            sims = int(sims)
        except (TypeError, ValueError) as exc:
            raise ValueError("Simulation count must be an integer.") from exc
        if not 1 <= sims <= MAX_SIMS:
            raise ValueError(f"Simulation count must be between 1 and {MAX_SIMS}.")

        model, metadata = self._load_model(checkpoint)
        engine = MCTS(model, self.device, num_sims=sims, batch_size=min(32, sims), c_puct=C_PUCT, temperature=0.0)

        started = time.perf_counter()
        with self._search_lock:
            root = engine.run(board=board, add_noise=False)
        elapsed = time.perf_counter() - started
        moves = list(root.children)
        if not moves:
            raise RuntimeError("The engine returned no legal moves.")
        ranked = sorted(moves, key=lambda move: root.children[move].visit_count, reverse=True)
        total_visits = sum(root.children[move].visit_count for move in moves) or 1

        candidates = []
        for rank, canonical_move in enumerate(ranked[:TOP_MOVES], start=1):
            child = root.children[canonical_move]
            actual_move = self._actual_move(canonical_move, board)
            candidates.append({
                "rank": rank,
                "san": board.san(actual_move),
                "uci": actual_move.uci(),
                "value": round(child.q_value, 4),
                "prior": round(child.prior, 4),
                "visits": child.visit_count,
                "visit_percent": round(100 * child.visit_count / total_visits, 2),
            })

        best = self._actual_move(ranked[0], board)
        return {
            "status": 200,
            "move": {"from": chess.square_name(best.from_square), "to": chess.square_name(best.to_square),
                     "promotion": chess.piece_symbol(best.promotion) if best.promotion else None,
                     "san": board.san(best)},
            # Keep the original flat fields for compatibility.
            "from": chess.square_name(best.from_square),
            "to": chess.square_name(best.to_square),
            "promotion": chess.piece_symbol(best.promotion) if best.promotion else None,
            "analysis": {
                "elapsed_seconds": round(elapsed, 3),
                "sims": sims,
                "checkpoint": checkpoint,
                "model": metadata,
                "top_moves": candidates,
                "lines": [
                    {"rank": index + 1, "moves": self._line(root, board, move)}
                    for index, move in enumerate(ranked[:TOP_LINES])
                ],
            },
        }

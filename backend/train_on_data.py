from __future__ import annotations

from dataclasses import dataclass

from board_encoder import board_to_tensor, move_to_action, NUM_ACTIONS
from backend.train_self_play import GameSample, ReplayBuffer, SampleDataset

@dataclass
class Config:
    # Network
    num_res_blocks: int = 4
    channels: int = 128
    
    # MCTS
    num_sims: int = 50
    
    # Training
    games_per_iter: int = 10
    max_game_moves: int = 200
    buffer_size: int = 10000
    batch_size: int = 128
    train_steps: int = 100
    lr: float = 1e-3
    num_iterations: int = 50
    checkpoint_every: int = 5
    
    # Misc
    device: str = "cuda"
    seed: int = 42


cfg = Config()

def chessbench_record_to_sample(
    record: dict,
    device: torch.device,
    tau: float = 0.1,
) -> GameSample:
    """
    Convert a ChessBench record into a GameSample.
    
    record format:
    {
        "fen": str,
        "moves": {
            "e2e4": {"win_prob": float},
            ...
        }
    }
    """
    import chess
    
    board = chess.Board(record["fen"])
    
    # --- 1. Encode board ---
    board_tensor = board_to_tensor(board, device=device).cpu()
    
    # --- 2. Build policy vector ---
    policy_vec = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    
    moves = []
    scores = []
    
    for move_uci, info in record["moves"].items():
        move = chess.Move.from_uci(move_uci)
        
        if move not in board.legal_moves:
            continue  # safety
        
        win_prob = info["win_prob"]
        
        # Convert win_prob → value in [-1, 1]
        value = 2 * win_prob - 1
        
        moves.append(move)
        scores.append(value)
    
    if len(scores) == 0:
        # fallback (shouldn’t happen often)
        return None
    
    scores = torch.tensor(scores, dtype=torch.float32)
    
    # --- 3. Softmax over moves ---
    probs = torch.softmax(scores / tau, dim=0)
    
    for move, p in zip(moves, probs):
        action = move_to_action(move)
        policy_vec[action] = p
    
    # --- 4. Compute value target ---
    # Expected value under the distribution
    outcome = torch.sum(probs * scores).item()
    
    return GameSample(board_tensor, policy_vec, outcome)

def load_chessbench_samples(
    filepath: str,
    max_samples: int,
    device: torch.device,
) -> list[GameSample]:
    import msgpack
    
    samples = []
    
    with open(filepath, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        
        for i, record in enumerate(unpacker):
            sample = chessbench_record_to_sample(record, device)
            
            if sample is not None:
                samples.append(sample)
            
            if len(samples) >= max_samples:
                break
    
    return samples

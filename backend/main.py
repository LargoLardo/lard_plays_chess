from flask import Flask, request
from flask_cors import CORS
import chess

import argparse

from network import ChessNet
from mcts import MCTS
from play import *
import time
# C:\Users\login\tree_fish\tree_fish\backend\venv\Scripts\activate.bat

# 500 sims for fast, weaker play
# 1600 sims for normal, stronger play

# python main.py checkpoint_iter4000.pt --color white --sims 500

# ------------ ENGINE LOGIC --------------

C_PUCT = 5.0
"""
(2.5) This solves the "shuffling problem" by making it actually search instead of 
relying too hard on its undertrained policy/value network 
"""

PROMOTION_MAP = {
    "q": chess.QUEEN,
    "r": chess.ROOK,
    "b": chess.BISHOP,
    "n": chess.KNIGHT,
    None: None
}   

board = chess.Board()

def resetBoard(fen: dict) -> dict:
    global board
    board = chess.Board(fen['fen'])
    return True

def makeResponse(move_dict: dict) -> dict:
    global board, mcts, endgame_engine, player_color
    from_square = chess.parse_square(move_dict['from'])
    to_square = chess.parse_square(move_dict['to'])
    promotion = PROMOTION_MAP.get(move_dict['promotion'])
    turn_color = chess.WHITE if move_dict['color'] == 'w' else chess.BLACK

    if str(board.piece_at(from_square)).lower() == 'p' and (chess.square_rank(to_square) == 0 or chess.square_rank(to_square) == 7):
        move = chess.Move(from_square, to_square, promotion)
    else:
        move = chess.Move(from_square, to_square)

    board.push(move)
    print(board)

    start_time = time.time()
    response = ai_move(board, mcts, endgame_engine)
    end_time = time.time()

    san = board.san(response)
    board.push(response)

    print(f"AI played:  {san}")
    print(f"AI processing time: {end_time - start_time:.2f} seconds\n")

    #Have some processing here
    return {
        "status": 200,
        "from": chess.square_name(response.from_square),
        "to": chess.square_name(response.to_square),
        "promotion": None if response.promotion is None else chess.piece_symbol(response.promotion)
    }

# ------------ ENDPOINT LOGIC --------------

app = Flask(__name__)
CORS(app)

@app.route("/send_move", methods=['PUT'])
def response_to_move():
    req = request.get_json()
    response = makeResponse(req)
    return response, 200

@app.route("/send_move/reset_board", methods=['PUT'])
def reset_board():
    req = request.get_json()
    response = resetBoard(req)
    if response:
        return "OK", 200
    else:
        return "Error resetting board", 500

# ------------ BACKEND STARTUP--------------

if __name__ == "__main__":
    global mcts, endgame_engine

    try:
        parser = argparse.ArgumentParser(description="Play against trained chess AI")
        parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
        parser.add_argument("--color", choices=["white", "black"], default="white",
                            help="Your color (default: white)")
        parser.add_argument("--sims", type=int, default=100,
                            help="MCTS simulations per move (default: 100)")
        parser.add_argument("--device", default="cuda",
                            help="Device: cuda or cpu (default: cuda)")
        parser.add_argument("--no-thinking", action="store_true",
                            help="Hide AI's candidate moves")
        args = parser.parse_args()
    except:
        raise Exception('Missing arguments, try using these: checkpoint_file.pt --color white --sims 50')

    ckpt, num_res_blocks, channels, iteration = load_checkpoint(args.checkpoint)

    player_color = chess.WHITE if args.color == "white" else chess.BLACK

    # Build network
    net = ChessNet(num_res_blocks=num_res_blocks, channels=channels)
    net.load_state_dict(ckpt["model"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    print(f"Loaded iteration {iteration} on {device}")
    print(f"Architecture: {num_res_blocks} res-blocks, {channels} channels\n")

    # Set up endgame engine
    endgame_engine = EndgameMinimax()

    # Set up MCTS
    mcts = MCTS(
        network=net,
        device=device,
        num_sims=args.sims,
        batch_size=32,
        c_puct=C_PUCT,
        temperature=0.0,         # deterministic best move
    )

    app.run(debug=True)
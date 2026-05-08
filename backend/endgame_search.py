import chess
import chess.polyglot
from dataclasses import dataclass

INF = 10**9
MATE_SCORE = 10**8
QUIESCENCE_MAX_DEPTH = 4

# -----------------------------
# Transposition Table Entry
# -----------------------------

@dataclass
class TTEntry:
    depth: int
    score: int
    flag: str       # "EXACT", "LOWER", "UPPER"
    best_move: chess.Move | None


# -----------------------------
# Endgame Minimax Engine
# -----------------------------

class EndgameMinimax:
    def __init__(self):
        self.tt = {}
        self.nodes = 0

        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }

        # Encourages king activity in endgames
        self.king_center_table = [
            -30, -20, -10, -10, -10, -10, -20, -30,
            -20, -5,   0,   0,   0,   0,  -5, -20,
            -10,  0,  10,  15,  15,  10,   0, -10,
            -10,  0,  15,  20,  20,  15,   0, -10,
            -10,  0,  15,  20,  20,  15,   0, -10,
            -10,  0,  10,  15,  15,  10,   0, -10,
            -20, -5,   0,   0,   0,   0,  -5, -20,
            -30, -20, -10, -10, -10, -10, -20, -30,
        ]

    # -----------------------------
    # Public search function
    # -----------------------------

    def search(self, board: chess.Board, depth: int = 4) -> tuple[chess.Move, int, int]:
        self.nodes = 0

        best_move = None
        best_score = -INF

        alpha = -INF
        beta = INF

        # Iterative deepening improves move ordering
        for current_depth in range(1, depth + 1):
            score, move = self.negamax(board, current_depth, alpha, beta, 0)

            if move is not None:
                best_move = move
                best_score = score

        return best_move, best_score, self.nodes

    # -----------------------------
    # Negamax with alpha-beta
    # -----------------------------

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int):
        terminal = self.terminal_score(board, ply)
        if terminal is not None:
            return terminal, None

        self.nodes += 1

        alpha_original = alpha
        key = chess.polyglot.zobrist_hash(board)

        # Terminal positions
        if board.is_checkmate():
            print("Checkmate detected at ply", ply)
            return -MATE_SCORE + ply, None

        if board.is_stalemate() or board.is_insufficient_material():
            return 0, None

        # Transposition table lookup
        if key in self.tt:
            entry = self.tt[key]

            if entry.depth >= depth:
                if entry.flag == "EXACT":
                    return entry.score, entry.best_move
                elif entry.flag == "LOWER":
                    alpha = max(alpha, entry.score)
                elif entry.flag == "UPPER":
                    beta = min(beta, entry.score)

                if alpha >= beta:
                    return entry.score, entry.best_move

        # Quiescence search at depth 0
        if depth == 0:
            return self.quiescence(board, alpha, beta, ply, qdepth=0), None

        best_score = -INF
        best_move = None

        moves = self.order_moves(board, list(board.legal_moves), self.tt.get(key))

        for move in moves:
            board.push(move)

            score, _ = self.negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                ply + 1
            )

            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

            if alpha >= beta:
                break

        # Store in transposition table
        if best_score <= alpha_original:
            flag = "UPPER"
        elif best_score >= beta:
            flag = "LOWER"
        else:
            flag = "EXACT"

        self.tt[key] = TTEntry(
            depth=depth,
            score=best_score,
            flag=flag,
            best_move=best_move
        )

        return best_score, best_move

    
    # -----------------------------
    # Quiescence search
    # Only searches captures/checks/promotions
    # -----------------------------

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int, qdepth: int):
        terminal = self.terminal_score(board, ply)
        if terminal is not None:
            return terminal

        self.nodes += 1

        stand_pat = self.evaluate(board)

        if qdepth >= QUIESCENCE_MAX_DEPTH:
            return stand_pat

        if stand_pat >= beta:
            return beta

        alpha = max(alpha, stand_pat)

        noisy_moves = []

        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion:
            # if board.is_capture(move) or board.gives_check(move) or move.promotion:
                noisy_moves.append(move)

        noisy_moves = self.order_moves(board, noisy_moves, None)

        for move in noisy_moves:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1, qdepth + 1)
            board.pop()

            if score >= beta:
                return beta

            alpha = max(alpha, score)

        return alpha

    # -----------------------------
    # Move ordering
    # -----------------------------

    def order_moves(self, board: chess.Board, moves, tt_entry=None):
        scored_moves = []

        tt_move = tt_entry.best_move if tt_entry else None

        for move in moves:
            score = 0

            # Try transposition table best move first
            if move == tt_move:
                score += 1_000_000

            # Captures: MVV-LVA
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)

                if victim and attacker:
                    score += 10_000
                    score += 10 * self.piece_values[victim.piece_type]
                    score -= self.piece_values[attacker.piece_type]

            # Promotions
            if move.promotion:
                score += 8_000 + self.piece_values.get(move.promotion, 0)

            # Checks
            if board.gives_check(move):
                score += 5_000

            # Prefer pushing passed-ish pawns in endgames
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                rank_gain = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)

                if piece.color == chess.WHITE:
                    score += rank_gain * 20
                else:
                    score -= rank_gain * 20

            scored_moves.append((score, move))

        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored_moves]

    # -----------------------------
    # Static evaluation
    # Positive means side to move is better
    # -----------------------------

    def evaluate(self, board: chess.Board):
        score = 0

        # Material
        for square, piece in board.piece_map().items():
            value = self.piece_values[piece.piece_type]

            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value

        # Endgame king activity
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)

        if white_king is not None:
            score += self.king_center_table[white_king]

        if black_king is not None:
            mirrored = chess.square_mirror(black_king)
            score -= self.king_center_table[mirrored]

        # Passed pawn bonus
        score += self.passed_pawn_score(board)

        # Mop-up bonus: if ahead, push enemy king to edge
        score += self.mop_up_score(board)

        # Convert from white perspective to side-to-move perspective
        return score if board.turn == chess.WHITE else -score

    # -----------------------------
    # Passed pawn scoring
    # -----------------------------

    def passed_pawn_score(self, board: chess.Board):
        score = 0

        for square, piece in board.piece_map().items():
            if piece.piece_type != chess.PAWN:
                continue

            if self.is_passed_pawn(board, square, piece.color):
                rank = chess.square_rank(square)

                if piece.color == chess.WHITE:
                    bonus = rank * rank * 10
                    score += bonus
                else:
                    bonus = (7 - rank) * (7 - rank) * 10
                    score -= bonus

        return score

    def is_passed_pawn(self, board: chess.Board, square: int, color: chess.Color):
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        enemy_color = not color

        files_to_check = [file]
        if file > 0:
            files_to_check.append(file - 1)
        if file < 7:
            files_to_check.append(file + 1)

        if color == chess.WHITE:
            ranks_to_check = range(rank + 1, 8)
        else:
            ranks_to_check = range(rank - 1, -1, -1)

        for f in files_to_check:
            for r in ranks_to_check:
                piece = board.piece_at(chess.square(f, r))
                if piece and piece.color == enemy_color and piece.piece_type == chess.PAWN:
                    return False

        return True

    # -----------------------------
    # Mop-up evaluation
    # Useful when one side is clearly winning
    # -----------------------------

    def mop_up_score(self, board: chess.Board):
        white_material = self.material_score(board, chess.WHITE)
        black_material = self.material_score(board, chess.BLACK)

        score = 0

        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)

        if white_king is None or black_king is None:
            return 0

        material_diff = white_material - black_material

        # If white is winning, push black king to edge
        if material_diff > 300:
            score += self.king_to_edge_bonus(black_king)
            score -= self.king_distance(white_king, black_king) * 10

        # If black is winning, push white king to edge
        elif material_diff < -300:
            score -= self.king_to_edge_bonus(white_king)
            score += self.king_distance(white_king, black_king) * 10

        return score

    def material_score(self, board: chess.Board, color: chess.Color):
        total = 0

        for piece in board.piece_map().values():
            if piece.color == color:
                total += self.piece_values[piece.piece_type]

        return total

    def terminal_score(self, board, ply):
        if board.is_checkmate():
            return -MATE_SCORE + ply

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        # if board.can_claim_draw():
        #     return 0

        return None

    def king_to_edge_bonus(self, king_square: int):
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)

        file_distance_to_edge = min(file, 7 - file)
        rank_distance_to_edge = min(rank, 7 - rank)

        distance_to_edge = min(file_distance_to_edge, rank_distance_to_edge)

        return (3 - distance_to_edge) * 30

    def king_distance(self, king_a: int, king_b: int):
        file_a = chess.square_file(king_a)
        rank_a = chess.square_rank(king_a)

        file_b = chess.square_file(king_b)
        rank_b = chess.square_rank(king_b)

        return max(abs(file_a - file_b), abs(rank_a - rank_b))
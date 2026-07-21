import unittest

import chess
import torch

from board_encoder import swap_move_color
from engine_service import EngineService
from mcts import MCTS, position_key
from network import ChessNet


class RepetitionHistoryTest(unittest.TestCase):
    def test_reconstructs_threefold_history(self):
        moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2
        board, counts = EngineService._board_from_history(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 8 5",
            moves,
        )
        self.assertTrue(board.is_repetition(3))
        self.assertEqual(counts[position_key(board)], 3)

    def test_rejects_history_that_does_not_match_fen(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            EngineService._board_from_history(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                ["e2e4"],
            )

    def test_mcts_marks_a_third_occurrence_as_terminal(self):
        moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]
        board, counts = EngineService._board_from_history(
            "rnbqkb1r/pppppppp/5n2/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 7 4",
            moves,
        )
        mcts = MCTS(ChessNet(num_res_blocks=1, channels=8), torch.device("cpu"), num_sims=0)
        root = mcts.run(board, repetition_counts=counts)
        repeating_move = swap_move_color(chess.Move.from_uci("f6g8"))
        self.assertEqual(root.children[repeating_move].repetition_count, 3)
        self.assertTrue(mcts._is_terminal(root.children[repeating_move]))

    def test_each_expansion_uses_one_network_call(self):
        class CountingNet(ChessNet):
            calls = 0

            def forward(self, inputs):
                self.calls += 1
                return super().forward(inputs)

        network = CountingNet(num_res_blocks=1, channels=8)
        MCTS(network, torch.device("cpu"), num_sims=1, batch_size=1).run(chess.Board())
        self.assertEqual(network.calls, 2)  # Root expansion plus one leaf expansion.


if __name__ == "__main__":
    unittest.main()

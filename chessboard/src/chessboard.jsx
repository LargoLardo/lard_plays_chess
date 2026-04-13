import React, { useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";

const ChessBoard = () => {
  const [game, setGame] = useState(new Chess());
  const [selectedSquare, setSelectedSquare] = useState(null);

  function makeMove(move) {
    const gameCopy = new Chess(game.fen());
    const result = gameCopy.move(move);

    if (!result) return false;

    setGame(gameCopy);
    return true;
  }

  const chessboardOptions = {
    position: game.fen(),
    arePiecesDraggable: true,

    onPieceDrop: ({ sourceSquare, targetSquare }) => {
      return makeMove({
        from: sourceSquare,
        to: targetSquare,
        promotion: "q",
      });
    },

    onSquareClick: (square) => {
      if (!selectedSquare) {
        const piece = game.get(square);
        if (piece && piece.color === game.turn()) {
          setSelectedSquare(square);
        }
        return;
      }

      makeMove({
        from: selectedSquare,
        to: square,
        promotion: "q",
      });

      setSelectedSquare(null);
    },

    /** Highlight selected square */
    customSquareStyles: selectedSquare
      ? {
          [selectedSquare]: {
            backgroundColor: "rgba(255, 200, 0, 0.4)",
          },
        }
      : {},
  };

  return <Chessboard options={chessboardOptions} />;
};

export default ChessBoard;
``
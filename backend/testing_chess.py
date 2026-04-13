import chess
import chess.svg
from IPython.display import display, SVG
import webbrowser, pathlib
path = pathlib.Path("board.svg")
path.write_text(chess.svg.board(), encoding="utf-8")
webbrowser.open(path.resolve().as_uri())

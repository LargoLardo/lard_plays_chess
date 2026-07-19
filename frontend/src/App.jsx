import { useCallback, useEffect, useState } from 'react'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import './App.css'
import { getCheckpoints, requestEngineMove, uploadCheckpoint } from './move_to_flask.js'

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

function outcome(game) {
  if (game.isCheckmate()) return game.turn() === 'w' ? 'Black wins by checkmate.' : 'White wins by checkmate.'
  if (game.isDraw()) return 'Game drawn.'
  return null
}

function App() {
  const [game, setGame] = useState(() => new Chess(START_FEN))
  const [checkpoints, setCheckpoints] = useState([])
  const [checkpoint, setCheckpoint] = useState('')
  const [sims, setSims] = useState(800)
  const [playerColor, setPlayerColor] = useState('w')
  const [selectedSquare, setSelectedSquare] = useState(null)
  const [thinking, setThinking] = useState(false)
  const [error, setError] = useState('')
  const [status, setStatus] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [timings, setTimings] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadsEnabled, setUploadsEnabled] = useState(false)

  const refreshCheckpoints = useCallback(async (preferred = '') => {
    const data = await getCheckpoints()
    setCheckpoints(data.checkpoints)
    setUploadsEnabled(Boolean(data.uploads_enabled))
    setCheckpoint((current) => {
      const wanted = preferred || current
      if (data.checkpoints.some((item) => item.name === wanted)) return wanted
      return data.checkpoints.at(-1)?.name || ''
    })
  }, [])

  useEffect(() => {
    refreshCheckpoints().catch((err) => setError(`Backend unavailable: ${err.message}`))
  }, [refreshCheckpoints])

  const runEngine = useCallback(async (position, chosenCheckpoint = checkpoint, chosenSims = sims) => {
    if (!chosenCheckpoint) throw new Error('Upload or select a checkpoint first.')
    setThinking(true)
    setError('')
    try {
      const response = await requestEngineMove(position.fen(), chosenCheckpoint, chosenSims)
      const next = new Chess(position.fen())
      next.move(response.move)
      setGame(next)
      setAnalysis(response.analysis)
      setTimings((items) => [...items, {
        ply: next.history().length,
        move: response.move.san,
        seconds: response.analysis.elapsed_seconds,
      }])
      setStatus(outcome(next) || '')
      return next
    } finally {
      setThinking(false)
    }
  }, [checkpoint, sims])

  async function startGame() {
    const fresh = new Chess(START_FEN)
    setGame(fresh)
    setSelectedSquare(null)
    setAnalysis(null)
    setTimings([])
    setStatus('')
    setError('')
    if (playerColor === 'b') {
      try { await runEngine(fresh) } catch (err) { setError(err.message) }
    }
  }

  async function makeMove(move) {
    if (thinking || game.turn() !== playerColor) return false
    const next = new Chess(game.fen())
    try {
      next.move(move)
      setGame(next)
      setSelectedSquare(null)
      const result = outcome(next)
      if (result) {
        setStatus(result)
      } else {
        await runEngine(next)
      }
      return true
    } catch (err) {
      const piece = game.get(move.to)
      setSelectedSquare(piece?.color === game.turn() ? move.to : null)
      if (!String(err.message).toLowerCase().includes('invalid move')) setError(err.message)
      return false
    }
  }

  function onSquareClick({ square }) {
    if (thinking || game.turn() !== playerColor) return
    if (!selectedSquare) {
      if (game.get(square)?.color === game.turn()) setSelectedSquare(square)
      return
    }
    makeMove({ from: selectedSquare, to: square, promotion: 'q' })
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError('')
    try {
      const result = await uploadCheckpoint(file)
      await refreshCheckpoints(result.checkpoint.name)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

  const boardOptions = {
    position: game.fen(),
    boardOrientation: playerColor === 'w' ? 'white' : 'black',
    arePiecesDraggable: !thinking && game.turn() === playerColor,
    onPieceDrop: ({ sourceSquare, targetSquare }) => makeMove({ from: sourceSquare, to: targetSquare, promotion: 'q' }),
    onSquareClick,
    squareStyles: selectedSquare ? { [selectedSquare]: { backgroundColor: 'rgba(126, 231, 135, .42)' } } : {},
  }

  return (
    <main className="app-shell">
      <header>
        <div><span className="eyebrow">POLICY + VALUE MCTS</span><h1>Tree Fish</h1></div>
        <div className={`engine-state ${thinking ? 'thinking' : ''}`}><i />{thinking ? 'ENGINE THINKING' : 'ENGINE READY'}</div>
      </header>

      <section className="control-bar">
        <label>Checkpoint
          <select value={checkpoint} onChange={(event) => setCheckpoint(event.target.value)} disabled={thinking}>
            {!checkpoints.length && <option value="">No checkpoints</option>}
            {checkpoints.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}
          </select>
        </label>
        <label>Simulations
          <input type="number" min="1" max="5000" value={sims} disabled={thinking}
            onChange={(event) => setSims(Math.max(1, Math.min(5000, Number(event.target.value) || 1)))} />
        </label>
        <label>Play as
          <select value={playerColor} onChange={(event) => setPlayerColor(event.target.value)} disabled={thinking}>
            <option value="w">White</option><option value="b">Black</option>
          </select>
        </label>
        <button className="primary" onClick={startGame} disabled={thinking || !checkpoint}>New game</button>
        {uploadsEnabled && (
          <label className="upload-button">{uploading ? 'Uploading…' : 'Upload .pt'}
            <input type="file" accept=".pt,.pth" onChange={handleUpload} disabled={uploading || thinking} />
          </label>
        )}
      </section>

      {(error || status) && <div className={error ? 'notice error' : 'notice'}>{error || status}</div>}

      <section className="workspace">
        <div className="board-panel">
          <Chessboard options={boardOptions} />
          <p className="turn-label">{status || (thinking ? 'Tree Fish is calculating…' : game.turn() === playerColor ? 'Your move' : 'Engine move')}</p>
        </div>

        <aside className="analysis-panel">
          <div className="panel-heading"><div><span className="eyebrow">LATEST SEARCH</span><h2>Engine analysis</h2></div>
            <strong>{analysis ? `${analysis.elapsed_seconds.toFixed(2)}s` : '—'}</strong></div>

          <div className="lines">
            <h3>Top engine lines</h3>
            {analysis?.lines?.length ? analysis.lines.map((line) => (
              <div className="line" key={line.rank}><span>{line.rank}</span><code>{line.moves.join(' ') || '—'}</code></div>
            )) : <p className="empty">Play a move to generate principal variations.</p>}
          </div>

          <div className="moves-table">
            <h3>Top ten moves</h3>
            <div className="table-row table-head"><span>#</span><span>Move</span><span>Value</span><span>Visits</span></div>
            {analysis?.top_moves?.map((move) => (
              <div className="table-row" key={move.uci}><span>{move.rank}</span><b>{move.san}</b><span className={move.value >= 0 ? 'positive' : 'negative'}>{move.value >= 0 ? '+' : ''}{move.value.toFixed(3)}</span><span>{move.visits} <small>{move.visit_percent}%</small></span></div>
            ))}
          </div>

          <div className="timings">
            <h3>Time per engine move</h3>
            {!timings.length && <p className="empty">No engine moves yet.</p>}
            {timings.map((item, index) => <div className="timing" key={`${item.ply}-${index}`}><span>{index + 1}. {item.move}</span><b>{item.seconds.toFixed(2)}s</b></div>)}
          </div>
        </aside>
      </section>
    </main>
  )
}

export default App

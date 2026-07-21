import { useCallback, useEffect, useMemo, useState } from 'react'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import './App.css'
import { getCheckpoints, requestEngineMove, uploadCheckpoint } from './move_to_flask.js'

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
const NEW_TIMELINE = [{ fen: START_FEN, move: null, analysis: null }]

function normalizeMove(move) {
  return { from: move.from, to: move.to, ...(move.promotion && { promotion: move.promotion }) }
}

function replay(timeline, cursor = timeline.length - 1) {
  const game = new Chess(START_FEN)
  for (const entry of timeline.slice(1, cursor + 1)) game.move(entry.move)
  return game
}

function uciHistory(timeline) {
  return timeline.slice(1).map(({ move }) => `${move.from}${move.to}${move.promotion || ''}`)
}

function outcome(game) {
  if (game.isThreefoldRepetition()) return 'Draw by threefold repetition.'
  if (game.isCheckmate()) return game.turn() === 'w' ? 'Black wins by checkmate.' : 'White wins by checkmate.'
  if (game.isDraw()) return 'Game drawn.'
  return ''
}

function App() {
  const [timeline, setTimeline] = useState(NEW_TIMELINE)
  const [cursor, setCursor] = useState(0)
  const [checkpoints, setCheckpoints] = useState([])
  const [checkpoint, setCheckpoint] = useState('')
  const [sims, setSims] = useState(800)
  const [playerColor, setPlayerColor] = useState('w')
  const [selectedSquare, setSelectedSquare] = useState(null)
  const [thinking, setThinking] = useState(false)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadsEnabled, setUploadsEnabled] = useState(false)

  const game = useMemo(() => replay(timeline, cursor), [timeline, cursor])
  const status = outcome(game)
  const analysis = timeline[cursor]?.analysis || null
  const timings = timeline.slice(0, cursor + 1).filter((entry) => entry.analysis)

  const refreshCheckpoints = useCallback(async (preferred = '') => {
    const data = await getCheckpoints()
    const ordered = [...data.checkpoints].sort((a, b) => Number(a.name.match(/checkpoint_iter(\d+)/)?.[1] || 0) - Number(b.name.match(/checkpoint_iter(\d+)/)?.[1] || 0))
    setCheckpoints(ordered)
    setUploadsEnabled(Boolean(data.uploads_enabled))
    setCheckpoint((current) => {
      const wanted = preferred || current
      if (ordered.some((item) => item.name === wanted)) return wanted
      return ordered.at(-1)?.name || ''
    })
  }, [])

  useEffect(() => {
    refreshCheckpoints().catch((err) => setError(`Backend unavailable: ${err.message}`))
  }, [refreshCheckpoints])

  const runEngine = useCallback(async (baseTimeline, chosenCheckpoint = checkpoint, chosenSims = sims) => {
    if (!chosenCheckpoint) throw new Error('Upload or select a checkpoint first.')
    setThinking(true)
    setError('')
    try {
      const position = replay(baseTimeline)
      const response = await requestEngineMove(
        position.fen(), chosenCheckpoint, chosenSims, uciHistory(baseTimeline),
      )
      const played = position.move(response.move)
      const updated = [...baseTimeline, {
        fen: position.fen(),
        move: normalizeMove(played),
        analysis: response.analysis,
        engineSan: played.san,
      }]
      setTimeline(updated)
      setCursor(updated.length - 1)
    } finally {
      setThinking(false)
    }
  }, [checkpoint, sims])

  async function startGame() {
    setTimeline(NEW_TIMELINE)
    setCursor(0)
    setSelectedSquare(null)
    setError('')
    if (playerColor === 'b') {
      try { await runEngine(NEW_TIMELINE) } catch (err) { setError(err.message) }
    }
  }

  async function makeMove(move) {
    if (thinking || status || game.turn() !== playerColor) return false
    const branch = timeline.slice(0, cursor + 1)
    const next = replay(branch)
    let played
    try {
      played = next.move(move)
    } catch (err) {
      const piece = game.get(move.to)
      setSelectedSquare(piece?.color === game.turn() ? move.to : null)
      if (!String(err.message).toLowerCase().includes('invalid move')) setError(err.message)
      return false
    }

    const updated = [...branch, { fen: next.fen(), move: normalizeMove(played), analysis: null }]
    setTimeline(updated)
    setCursor(updated.length - 1)
    setSelectedSquare(null)
    if (!outcome(next)) {
      try { await runEngine(updated) } catch (err) { setError(err.message) }
    }
    return true
  }

  function onSquareClick({ square }) {
    if (thinking || status || game.turn() !== playerColor) return
    if (!selectedSquare) {
      if (game.get(square)?.color === game.turn()) setSelectedSquare(square)
      return
    }
    makeMove({ from: selectedSquare, to: square, promotion: 'q' })
  }

  function navigate(nextCursor) {
    setCursor(nextCursor)
    setSelectedSquare(null)
    setError('')
  }

  function downloadPgn() {
    const pgnGame = replay(timeline, cursor)
    pgnGame.header(
      'Event', 'Tree Fish Web Game',
      'White', playerColor === 'w' ? 'Player' : 'Tree Fish',
      'Black', playerColor === 'b' ? 'Player' : 'Tree Fish',
      'Result', pgnGame.isCheckmate() ? (pgnGame.turn() === 'w' ? '0-1' : '1-0') : pgnGame.isDraw() ? '1/2-1/2' : '*',
    )
    const url = URL.createObjectURL(new Blob([pgnGame.pgn()], { type: 'application/x-chess-pgn' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `tree-fish-ply-${cursor}.pgn`
    link.click()
    URL.revokeObjectURL(url)
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
    arePiecesDraggable: !thinking && !status && game.turn() === playerColor,
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
            {checkpoints.map((item, index) => <option key={item.name} value={item.name}>Strength Level {index + 1}</option>)}
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
          <div className="history-controls">
            <button onClick={() => navigate(cursor - 1)} disabled={thinking || cursor === 0}>← Back</button>
            <span>Ply {cursor} / {timeline.length - 1}</span>
            <button onClick={() => navigate(cursor + 1)} disabled={thinking || cursor === timeline.length - 1}>Forward →</button>
            <button onClick={downloadPgn} disabled={thinking}>Download PGN</button>
          </div>
          <p className="turn-label">{status || (thinking ? 'Tree Fish is calculating…' : game.turn() === playerColor ? 'Your move' : 'Engine move')}</p>
        </div>

        <aside className="analysis-panel">
          <div className="panel-heading"><div><span className="eyebrow">SEARCH AT THIS PLY</span><h2>Engine analysis</h2></div>
            <strong>{analysis ? `${analysis.elapsed_seconds.toFixed(2)}s` : '—'}</strong></div>

          <div className="lines">
            <h3>Top engine lines</h3>
            {analysis?.lines?.length ? analysis.lines.map((line) => (
              <div className="line" key={line.rank}><span>{line.rank}</span><code>{line.moves.join(' ') || '—'}</code></div>
            )) : <p className="empty">No engine search is cached at this position.</p>}
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
            {timings.map((entry, index) => <div className="timing" key={index}><span>{index + 1}. {entry.engineSan}</span><b>{entry.analysis.elapsed_seconds.toFixed(2)}s</b></div>)}
          </div>
        </aside>
      </section>
    </main>
  )
}

export default App

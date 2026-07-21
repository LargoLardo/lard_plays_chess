const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`)
  return body
}

export function getCheckpoints() {
  return api('/checkpoints')
}

export function requestEngineMove(fen, checkpoint, sims, moves) {
  return api('/engine/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fen, checkpoint, sims, moves }),
  })
}

export function uploadCheckpoint(file) {
  const form = new FormData()
  form.append('checkpoint', file)
  return api('/checkpoints', { method: 'POST', body: form })
}

# Tree Fish web app

Tree Fish is a React chess client backed by a PyTorch policy/value network and
Monte Carlo tree search. The browser keeps the game state and sends the current
FEN, selected checkpoint, and simulation count to the API. The API returns the
engine move plus three principal variations, ten ranked candidates, their Q
values/priors/visits, and search time.

## Run locally

Requirements: Python 3.10+, Node 20+, and enough RAM to load a checkpoint. Use a
CUDA-enabled PyTorch install for GPU inference; the API falls back to CPU.

From one terminal:

```powershell
python -m venv backend/venv
backend/venv/Scripts/Activate.ps1
pip install -r backend/requirements-api.txt
python backend/main.py --checkpoint-dir checkpoints --debug
```

From a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). Vite proxies API calls to
Flask at `http://localhost:5000`. Existing `.pt` files in `checkpoints/` appear
in the selector. You can also upload a weights-only `.pt`/`.pth` checkpoint in
the UI. It must contain `model`, and should contain `num_res_blocks`, `channels`,
and `iteration`.

## Deploy the backend to Modal

The deployment uses a T4 GPU and a persistent Modal Volume. The volume stores
checkpoints independently from the container, while each warm container caches
the two most recently selected models. The deployment is intentionally limited
to one container/one request at a time because a checkpoint is large and each
MCTS search consumes the GPU. Modal scales it to zero after five idle minutes.

```powershell
pip install modal
modal setup
modal secret create tree-fish-config CORS_ORIGINS="*" MAX_SIMS=5000 MAX_CHECKPOINT_BYTES=786432000
modal volume create tree-fish-checkpoints
modal volume put tree-fish-checkpoints checkpoints/checkpoint_iter6000.pt /checkpoint_iter6000.pt
modal deploy backend/modal_app.py
```

The volume creation command is optional because the deployment creates it when
missing. `modal deploy` prints the public API URL. Test it with:

```powershell
curl.exe https://YOUR-MODAL-URL.modal.run/health
```

You can add later checkpoints either with the app's **Upload .pt** control or
with `modal volume put`. The browser upload validates the file before committing
it to the persistent volume. For very large files, the CLI upload is generally
more reliable. If using the CLI while a container is warm, redeploy/restart the
app so its mounted volume view refreshes.

The `tree-fish-config` Modal Secret supplies `MAX_SIMS`,
`MAX_CHECKPOINT_BYTES`, and `CORS_ORIGINS` to Flask. Replace `*` with your Vercel
production URL if you do not want public CORS; for preview deployments, leave it
as `*` or supply all allowed origins as a comma-separated value. `TORCH_DEVICE`
can also be added to the secret when you need to override device selection.

## Deploy the frontend to Vercel

1. Import this repository in Vercel.
2. Set **Root Directory** to `frontend`.
3. Add `VITE_API_BASE_URL` in Project Settings → Environment Variables and set
   it to the Modal URL, with no trailing slash.
4. Deploy. `vercel.json` selects Vite, runs `npm run build`, and serves `dist`.

For CLI deployment, after setting the environment variable in Vercel:

```powershell
cd frontend
npm install -g vercel
vercel
vercel --prod
```

Vite substitutes `VITE_API_BASE_URL` at build time, so changing it requires a
new frontend deployment.

## API design

- `GET /health` reports API/device readiness.
- `GET /checkpoints` lists playable checkpoint files.
- `POST /checkpoints` accepts multipart field `checkpoint`.
- `POST /engine/move` accepts `{ "fen", "checkpoint", "sims" }`.

There is no server-side global chessboard. That is important on Modal: requests
may arrive after a cold start, and one user's game must not overwrite another's.
Candidate `value` is the MCTS child's average Q value in `[-1, 1]`; it is a model
evaluation, not a Stockfish centipawn score. Visits show how the simulation
budget was allocated. The displayed lines follow the most-visited continuation
already present in the search tree, up to five plies.

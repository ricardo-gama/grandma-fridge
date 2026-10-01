# Avó's Fridge Rescue

An LLMOps class project: tell Avó what's in your fridge, and she turns it into a recipe — in one of four grandma personas. Built with Docker, MLflow (experiment tracking, tracing, and a Prompt Registry), Gemini (with a local Ollama fallback), and a Flask API that also serves its own frontend — the whole app is one container, reachable at `http://localhost:8000`, nothing external required.

The project follows the same discipline as the course's own example: prompts are versioned like models, scored against a fixed test set, and only promoted to serve live traffic after clearing a quality gate.

| | |
|---|---|
| The thing you version | a grandma **persona** (prompt) |
| Where versions live | MLflow **Prompt Registry** |
| How you compare them | a fixed **Evaluation Set** |
| How you ship one | move the `@champion` alias |
| What serves it | the Flask API, at `http://localhost:8000` — UI and API together |

## What you need

- Docker Desktop (with Docker Compose)
- A free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no card required
- ~4GB free disk space if you also want the optional local-LLM fallback (Ollama)

## Quick start

```bash
git clone https://github.com/<your-username>/grandma-fridge.git
cd grandma-fridge

cp docker/.env.example docker/.env
# then open docker/.env and paste in your GEMINI_API_KEY

docker compose -f docker/docker-compose.yml up -d --build
```

Open **http://localhost:8000** — that's the whole app: pick a persona, type or click a quick-idea suggestion, hit "Ask Avó."

| Service | URL | What it is |
|---|---|---|
| Avó's app | http://localhost:8000 | The frontend *and* the API — same container, same port |
| MLflow | http://localhost:5001 | Experiments, traces, and the Prompt Registry |
| JupyterLab | http://localhost:8888/?token=avo | Where personas are prototyped and evaluated |

Or hit the API directly:

```bash
curl -X POST http://localhost:8000/recipe \
  -H "Content-Type: application/json" \
  -d '{"fridge_items": "ovos, bacalhau, batata, cebola, azeite"}'
```

Ask about something that isn't food and watch Avó refuse — that's a feature, not a bug, and it's one of the things the evaluation measures.

## The cycle

### 1. Four personas

[`src/grandma_personas.py`](src/grandma_personas.py) holds four grandma voices:

| Persona | Approach |
|---|---|
| `affectionate` | Warm, encouraging, simple language |
| `strict` | Judges your fridge choices first, then precise steps |
| `dramatic` | Telenovela energy, everything is a crisis or a triumph |
| `practical` | No commentary, fastest possible recipe |

All four write mainly in English with Portuguese words woven in naturally (food names, endearments, exclamations) — this is shared across every persona via one `LANGUAGE_STYLE` constant, so the voice stays consistent.

This file is imported by both the evaluation pipeline and the live API — the persona being *scored* is byte-for-byte the persona being *served*.

### 2. The Evaluation Set

[`src/evaluation_set.py`](src/evaluation_set.py) is the fixed bar: real fridge lists with reference recipes, plus off-topic cases Avó must refuse — including two prompt-injection attempts ("ignore your instructions...", "forget you are a grandma..."). It is never used to fine-tune anything; it exists purely so personas can be compared on identical input.

### 3. Evaluate and promote

Evaluation runs from [`notebooks/grandma_prototyping.ipynb`](notebooks/grandma_prototyping.ipynb) — open JupyterLab at http://localhost:8888/?token=avo, open that notebook, and run the evaluation cell:

```python
import sys
sys.path.insert(0, "/home/jovyan/work")  # repo root as seen inside the jupyter container

import src.evaluate_personas as ep

sys.argv = ["evaluate_personas"]   # add "--promote" here once a persona clears the gate
ep.main()
```

This calls the same `main()` as the CLI would (`docker compose exec api python -m src.evaluate_personas`) — same scoring, same MLflow registration, same gate — the `sys.argv` override just keeps Jupyter's own kernel arguments from confusing argparse. Running it here means it uses the `jupyter` container's environment (`MLFLOW_TRACKING_URI`, `LLM_REQUESTS_PER_MINUTE`), and you get the results back as a Python object (`ep.main()`'s return value, or whatever `score_persona`/ranking variables the notebook exposes) to inspect or plot immediately, instead of only reading logs.

Each persona is registered as a version of the prompt `avo-fridge-persona`, scored over every case, and rated on:

| Metric | What it catches |
|---|---|
| `ingredient_usage` | Did the recipe use what's actually in the fridge? |
| `refusal_accuracy` | Did it refuse everything off-topic, including injection attempts? |
| `rougeL` | Word overlap with a reference recipe |
| `actionability` | Are there actual cooking steps, not just description? |
| `ollama_fallback_rate` | How often Gemini failed and Ollama answered instead |
| `truncated_responses` / `failed_calls` | Silent-failure tripwires |

Compare personas in MLflow at http://localhost:5001.

`refusal_accuracy` has a hard gate at 1.0. A persona that answers something it was told to refuse is **not promoted**, however well it scores elsewhere — a confident, on-brand, off-topic answer is worse than no answer.

Once a persona clears the gate, re-run the same cell with `--promote`:

```python
sys.argv = ["evaluate_personas", "--promote"]
ep.main()
```

then reload the live app:

```bash
curl -X POST http://localhost:8000/persona/reload
```

Avó now answers with the new champion — no rebuild, no redeploy.

> Prefer the command line instead? The equivalent, run from the `api` container rather than `jupyter`, is:
> ```bash
> docker compose -f docker/docker-compose.yml exec api python -m src.evaluate_personas
> docker compose -f docker/docker-compose.yml exec api python -m src.evaluate_personas --promote
> ```

### 4. The frontend

There's no separate frontend service. `api/fridge_app.py` serves the UI itself, as static files (`api/static/index.html`, `styles.css`, `app.js`), from the same Flask process that serves `/recipe`. That means:

- One container, one port (`8000`) — frontend and API are same-origin, so there's no CORS to configure and nothing external (no tunnel, no separate hosting) needed to run the whole app locally.
- The persona picker calls `/persona/switch`; the four quick-idea chips (bacalhau & eggs, chicken & rice, pasta & tomato, leftover odds & ends) are pulled straight from `evaluation_set.py`'s `ON_TOPIC` cases, so whatever a demo viewer clicks is something the evaluation pipeline has already scored.
- Each response shows which persona and which provider (`gemini` or `ollama`) actually answered — useful for showing the fallback working live, not just in a metric.

A design draft for this UI was first prototyped in Lovable; the shipped version here is a hand-built equivalent so the whole stack — UI included — runs as one Docker image rather than depending on Lovable's hosting or a tunnel. The `frontend/` folder from the original plan is retired in favor of this.

## Optional: local LLM fallback

`src/llm_client.py` tries Gemini first and silently falls back to a local Ollama model on any error (rate limit, quota, network). This is off by default; to include it:

```bash
docker compose -f docker/docker-compose.yml --profile local-llm up -d
docker compose -f docker/docker-compose.yml exec ollama ollama pull llama3.2:3b
```

A small model like `llama3.2:3b` needs ~2-4GB disk and 4-8GB RAM, runs CPU-only, and answers in a few seconds. Worth knowing: it's noticeably weaker than Gemini, so expect lower scores when the evaluation runs on the fallback.

## Layout
```
grandma-fridge/
├── .github/
│   └── workflows/
│       └── ci.yml              # stack rules + notebook hygiene, runs on every push/PR
├── tests/                      # static checks ci.yml runs — nothing has to be running
│   ├── test_compose.py
│   ├── test_dockerfiles.py
│   ├── test_requirements.py
│   ├── test_secrets.py
│   ├── test_personas.py
│   ├── test_evaluation_set.py
│   └── test_notebooks.py
├── docker/
│   ├── docker-compose.yml     # mlflow + jupyter + api (+ ollama, optional profile)
│   ├── Dockerfile.api
│   ├── Dockerfile.jupyter
│   ├── init-mlflow.sh
│   ├── requirements.txt       # pinned
│   └── .env.example
├── api/
│   ├── fridge_app.py           # Flask service — serves prompts:/avo-fridge-persona@champion AND the UI
│   └── static/
│       ├── index.html          # the frontend
│       ├── styles.css
│       └── app.js
├── src/
│   ├── llm_client.py          # the only file that knows Gemini (+ Ollama fallback)
│   ├── grandma_personas.py    # the four personas, shared by pipeline and service
│   ├── evaluation_set.py      # the fixed bar
│   └── evaluate_personas.py   # score, rank, gate, promote
└── notebooks/
    └── grandma_prototyping.ipynb  # persona prototyping AND evaluation — run it from here
```

## Continuous Integration

Every push and pull request (including from forks) runs `tests/` in GitHub Actions (`.github/workflows/ci.yml`): stack rules (compose services wired correctly, secrets never hardcoded, Dockerfiles pinned) and notebook hygiene (no committed cell outputs, no leaked keys, personas and evaluation set structurally intact). Nothing here calls Gemini or needs a secret, so every contributor — including a fork with no access to your API key — gets the same feedback in seconds.

## Future work

Scoped out for now, in rough priority order:

- **An on-topic/off-topic classifier.** `evaluation_set.py`'s `ON_TOPIC`/`OFF_TOPIC` cases are already labeled — a small classifier trained on them could sit in front of the LLM call as a cheap pre-filter (obviously junk input gets refused without spending a Gemini call), and would register to MLflow's **Model Registry**, alongside the Prompt Registry already in use.
- **A real Gemini-vs-Ollama comparison.** Today Ollama is only a fallback triggered on Gemini errors (`ollama_fallback_rate` counts *incidents*, not a controlled comparison). Running `score_persona()` once per provider, logged as separate MLflow runs, would show the actual quality/cost tradeoff.
- **Run the evaluation gate itself in CI**, not just static stack/notebook checks — a workflow that builds the images and calls `evaluate_personas.py` against a throwaway MLflow instance on every push, failing the build if `refusal_accuracy` or `overall_score` don't clear the bar. Held back deliberately: it needs a `GEMINI_API_KEY` secret, which a fork's CI run wouldn't have, so today's CI stays secret-free and fork-friendly instead.
- **A publicly reachable deployment** — today's setup (Docker Compose on one machine) is fully containerized and satisfies the course's "deploy in a container" requirement, but it depends on that machine staying on. Deploying `mlflow` + `api` together to a host like Railway or Render (same Dockerfiles, no rewrite) would make it reachable without a laptop running.

## Secrets

`docker/.env` is gitignored, and `docker-compose.yml` only ever names the variable that holds your key, never the key itself. Never commit it, never paste it into a chat window. Anyone who gets it can spend your Gemini quota.

## Troubleshooting

**Colleagues cloning this repo**: `docker/.env` will not exist after cloning (it's gitignored on purpose). Run `cp docker/.env.example docker/.env` and add your own Gemini key before starting.

**`Avó answers "I'm not configured right now"`** — `GEMINI_API_KEY` is missing from `docker/.env`. Check with:
```bash
curl -s http://localhost:8000/health
```

**Build fails on `pyarrow` / `pkg_resources`** — this happens if the Jupyter/API base image picks up Python 3.13, which some pinned dependencies don't have wheels for yet. The Dockerfiles here are pinned to Python 3.12 (`quay.io/jupyter/scipy-notebook:python-3.12`) for exactly this reason — if you've changed the base image tag, revert it.

**Port 5000 already in use** — on macOS this is almost always AirPlay Receiver. MLflow is mapped to host port **5001** here for that reason (`http://localhost:5001`, not 5000).

**`Invalid Host header - possible DNS rebinding attack detected`** — MLflow rejects Host headers it doesn't recognise. `init-mlflow.sh` passes `--allowed-hosts`; if you changed a service name or port mapping in `docker-compose.yml`, add the new host:port there too.

**Gate fails on `refusal_accuracy`** — expected behavior, not a bug. Check which specific off-topic case slipped through (see `src/evaluate_personas.py`'s `looks_like_refusal`), and either strengthen that persona's refusal instruction or accept that persona isn't ready to ship.

**`429 RESOURCE_EXHAUSTED` from Gemini** — you've hit the free tier's rate or daily quota limit. Either wait, lower `--rpm` on the evaluation script, or bring up the Ollama fallback (see above) — `llm_client.py` will use it automatically once available.

**Docker Desktop won't start (Windows)** — usually a WSL2 issue. Try `wsl --update` then `wsl --shutdown`, then reopen Docker Desktop. If disk space is the blocker, run Disk Cleanup (`cleanmgr`) targeting Windows Update files first.

**`avo-api` keeps restarting / `python: can't open file '/app/api/fridge_app.py'`** — `api/fridge_app.py` doesn't exist on your machine yet, or is incomplete (check with `wc -l api/fridge_app.py` — it should be a few hundred lines, ending cleanly with `app.run(...)`). Rebuild after fixing it: `docker compose -f docker/docker-compose.yml build --no-cache api && docker compose -f docker/docker-compose.yml up -d api`.

**CI fails on `test_no_committed_cell_outputs`** — a notebook was saved with run output still in it. Clear outputs before committing: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`, or `Kernel → Restart & Clear Output` in Jupyter.

**Want to share the running app with someone outside your machine?** Everything here is designed to run at `localhost:8000` with nothing external needed. If you do want it reachable from another device or over the internet temporarily (a remote demo, say), tunnel it: `ngrok http 8000`, then use the `https://....ngrok-free.app` URL it prints. That URL changes every time you restart the tunnel on the free tier — for anything longer-lived, see Future Work's deployment note above.

**Start over**

```bash
docker compose -f docker/docker-compose.yml down -v
rm -rf artifacts/
docker compose -f docker/docker-compose.yml up -d --build
```
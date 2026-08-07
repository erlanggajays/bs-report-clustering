# Test Execution & Analytics Engine — Phase 1

Dynamically ingests mobile app test results from BrowserStack for a target
**project**, applies ML/NLP to cluster failure stack traces, and renders a
self-contained interactive **static HTML report**.

> The owner/display name is optional: set `TARGET_USER` to override, otherwise
> it is derived from the build name and hidden in the report if unknown.

> Phase 1 scope: static HTML only. Slack bots, PR comments, and live dashboards
> (Streamlit) are **out of scope**.

## Project structure

```
test-analytics-engine/
├── pyproject.toml             # packaging, deps, entry point, ruff/mypy/pytest config
├── main.py                    # thin shim so `python main.py` still works
├── src/
│   ├── config.py              # settings + endpoints (creds from env only)
│   ├── cli.py                 # argparse + logging entry point (`bsa-report`)
│   ├── ingestor.py            # BrowserStack / mock ingestion -> DataFrame
│   ├── triage_engine.py       # TF-IDF + DBSCAN clustering, anomaly, flakiness
│   ├── exec_metrics.py        # suite health, MTTR saved, device risk matrix
│   ├── report_generator.py    # Plotly + Jinja2 -> standalone HTML
│   └── history.py             # SQLite cross-build history (flakiness + trend)
├── templates/report_template.html
├── scripts/generate_mock_data.py   # realistic mock payload (no creds needed)
├── tests/                     # pytest suite
├── .github/workflows/ci.yml   # ruff + mypy + pytest on push/PR
├── data/                      # mock JSON + SQLite history land here
└── output/                    # browserstack_report.html lands here
```

## Quick start (no credentials)

```bash
pip install -e .                              # or: pip install -r requirements.txt
python scripts/generate_mock_data.py          # writes data/mock_browserstack_build.json
python main.py                                # writes output/browserstack_report.html
# ...or, after `pip install -e .`, use the console script:
bsa-report
```

Open `output/browserstack_report.html` in any browser — it is fully standalone
(Plotly JS is embedded inline; no network required). Add `-v/--verbose` for DEBUG logs.

## Development

```bash
pip install -e ".[dev]"    # installs pytest, ruff, mypy
pytest                     # run the test suite
ruff check src tests       # lint
mypy src                   # type-check
```

CI (`.github/workflows/ci.yml`) runs all three across Python 3.10–3.12 on every push/PR.

## Platforms: separate or combined

Default is **one project per run** — a clean per-platform CI gate:

```bash
python main.py --source api --project "Finserv - Gopay Android"
python main.py --source api --project "Finserv - Gopay iOS"
```

Repeat `--project` (or pass a comma-separated list) for a **cross-platform report**,
which makes `platform` an attribution dimension:

```bash
python main.py --source api \
  --project "Finserv - Gopay Android" --project "Finserv - Gopay iOS"
```

Combined mode adds a **platform breakdown** (always shown separately, so one
platform's regression can never hide behind the other's green) and a
**business area × platform** table. A gap for the same area between platforms
points to a platform-specific defect; a high rate on both points to the product.

> `platform` is read from each session's `os` field, so it needs no configuration.
> Note the taxonomy rules are currently Android-flavoured (uiautomator2/ADB
> vocabulary); iOS failures will land in `uncategorized` until iOS rules are added.

## Analysis modes

| Mode | Command | What it analyzes |
|---|---|---|
| **Latest** (default) | `main.py --source api` | The newest completed build — a CI-gate / smoke view |
| **Range** | `main.py --source api --mode range --last 20` | The last N builds aggregated — clusters, device risk and MTTR gain real statistical power |

Range mode (`--source api`) walks the last N eligible builds from
`projects/{id}.json`, fetches + persists each build's sessions (capped at
`max_range_builds`, default 25), then analyzes the combined set. Flakiness uses
the true cross-build pass↔fail flip rate. `--source file --mode range` aggregates
the seeded sample history DB for local testing.

## Live mode

Selection is **project-driven**. The pipeline resolves the project, picks its
newest completed build (`done` / `failed` / `timeout`), then fetches its sessions:

```
projects.json  ──▶  projects/{id}.json (nested builds[])  ──▶  builds/{hashed_id}/sessions.json
   name→id            newest build in BUILD_STATUS_FILTER       session rows → DataFrame
```

Credentials are read **only** from the environment — never hardcode them:

```bash
export BROWSERSTACK_USERNAME=...    # or load from your secrets vault
export BROWSERSTACK_ACCESS_KEY=...
python main.py --source api --project "Finserv - Gopay Android"
```

Host: `https://api-cloud.browserstack.com/app-automate`.

| Env var / flag | Purpose | Default |
|---|---|---|
| `TARGET_PROJECT` / `--project` | Project name to resolve | `Finserv - Gopay Android` |
| `TARGET_PROJECT_ID` | Skip name→id lookup (e.g. `2256311`) | *(unset)* |
| `BUILD_STATUS_FILTER` | Eligible build statuses, comma-sep | `done,failed,timeout` |
| `ENRICH_LOGS` / `--enrich-logs` | Pull terminal logs for failed sessions to recover real stack traces | off |
| `HISTORY_DB_PATH` | SQLite file for cross-build history | `data/history.db` |

Sample (`--source file`) runs persist to a **separate** `data/history_sample.db`
so demo data never pollutes real history. Seed demo history with
`python scripts/generate_mock_data.py --seed-history 8`.

> **On `reason`:** BrowserStack's `sessions.json` sets `reason` to `"COMPLETED"`
> for passing sessions; real Appium stack traces live in the terminal logs.
> Enable `--enrich-logs` for meaningful failure clustering.

## Report contents

- **Statistical findings** — associations that clear significance (Fisher's exact,
  one level vs rest) with odds ratio + 95% CI. Confounded dimensions (a device that
  maps 1:1 to an OS version) are collapsed to one finding rather than double-counted.
- **Where failures concentrate** — failure rate by **business area** (derived from
  test names), **locator hotspots** (one broken locator often explains many tests),
  and **runtime outliers** (median/MAD, robust to the outlier itself).
- **Executive cards** — total tests, suite health, pass rate, root causes, MTTR saved.
- **Pass/fail donut** + **device/OS heatmap** (Wilson-scored, small samples flagged).
- **Suite pass-rate trend** across historical builds.
- **Failure categories** — rule-based taxonomy routing each failure to an **owner**
  (Dev / Backend / Test automation / Infra), so you see app-bugs vs test-flakes vs infra.
- **Failure clusters** — signature fingerprint (`Exception @ Class.method`),
  affected app version, confidence, and a **deep-link to the session replay**.
- **Flaky scenarios** — true pass↔fail flip-rate once history accumulates.

## Failure taxonomy

Failures are classified into named categories with an owner. Rules come from code
defaults, optionally **overridden by `config/taxonomy.json`** (evaluated first — see
`config/taxonomy.example.json` for the format; set `FAILURE_TAXONOMY_PATH` to relocate).
A category matches on regex over `reason` + log text, session status, or a duration
threshold (`MIN_VALID_TEST_SECONDS`, default 60 → "test did not run"). Anything
unmatched falls through to the fingerprint clustering.

Native-dialog / crash categories only fire when the logs are available, so run
with `--enrich-logs`. Enrichment fetches the configured **log sources** per failed
session (`LOG_SOURCES`, default `crash,appium`; also `device`, `text`) — feeding
both the classifier and stack-trace clustering:

- **`crash`** (`/crashlogs`) — a present crashlog reliably tags the failure as
  `app-crash` (owner: Dev), even when the surface symptom looks like element-not-found.
- **`appium`** (`/appiumlogs`) — richest detail: permission dialogs, failed locators,
  WebView context errors.
- **`device`** (`/devicelogs`) — full logcat (large; opt-in) for ANRs / native issues.

More sources = fewer `uncategorized`, at the cost of extra HTTP calls per failed session.

## Design notes

- **Clustering:** TF-IDF (word n-grams) + DBSCAN with cosine distance. DBSCAN
  discovers the number of root causes automatically and isolates one-off
  failures; the vectorizer is swappable for sentence-embeddings later.
- **Flakiness:** in-build proxy (status variance + duration CoV). Wire in
  historical builds for true run-over-run flakiness in a later phase.
- **MTTR saved:** `(failures − clusters) × manual_triage_minutes_per_failure`,
  configurable in `config.py`.
```

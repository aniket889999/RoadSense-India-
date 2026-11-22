# Running the ANPR API locally or in a container

RoadSense is an experimental, local review tool. These options make it easier to run
and safer to expose; they do not make it a production service.

## Local (recommended for the demo)

```bash
.venv/bin/python scripts/anpr_preflight.py     # read-only readiness check
.venv/bin/python scripts/serve_anpr_local.py   # RapidOCR review profile, MPS detector
```

## Optional access controls (off by default)

| Variable | Effect when set |
| --- | --- |
| `ROADSENSE_API_KEY` | `POST /api/v1/anpr/images/recognize` requires header `X-API-Key` with this value (constant-time comparison); otherwise 401 |
| `ROADSENSE_ANPR_RATE_LIMIT_PER_MINUTE` | at most N requests per client address per rolling minute; otherwise 429 with `Retry-After` |

Unset, the endpoint behaves exactly as before. The rate limiter keeps counts in process
memory: it is per worker, resets on restart and is not shared between instances. Never
put the API key in browser code; it is meant for scripts or a reverse proxy calling the
API. The key, images and plate text are not logged.

## Container (CPU)

```bash
docker compose -f infra/docker-compose.yml --profile api up --build api
```

- The `api` service is opt-in (profile `api`); a plain `docker compose up` still starts
  only the existing database service.
- The image contains code and the public RTO directory only. Models are mounted
  read-only from `./models`; `./outputs` is mounted for the local SQLite database.
  Nothing is downloaded at runtime.
- Docker on macOS cannot use Apple MPS, so the plate detector runs on CPU in the
  container. Measured warm latency with a CPU detector on the M4 host was similar to MPS
  (median ≈ 0.61 s per image; see `docs/anpr/latency_results.md`); container overhead
  has not been measured.
- The port is published on `127.0.0.1` only. Set the access controls above before
  publishing it more widely.
- The image build itself has not been run as part of this change; the configuration is
  validated by `tests/test_ci_container_config.py`.

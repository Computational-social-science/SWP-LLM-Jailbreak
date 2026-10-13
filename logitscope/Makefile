.PHONY: setup server server-tiny web test test-server test-web e2e bench shots dev

setup:
	cd server && uv sync
	cd web && npm install

# real model (downloads smollm2 weights on first session)
server:
	cd server && .venv/bin/python -m logitscope --port 8000

# instant startup, random weights — for poking at the ui
server-tiny:
	cd server && .venv/bin/python -m logitscope --tiny --port 8000

web:
	cd web && npm run dev

# both at once: tiny server in the background, vite in the foreground
dev:
	cd server && .venv/bin/python -m logitscope --tiny --port 8000 & \
	cd web && npm run dev

test-server:
	cd server && .venv/bin/pytest -q

test-web:
	cd web && npm test

test: test-server test-web

# playwright against the committed replay fixture — no weights needed
e2e:
	cd web && npx playwright test

# regenerate docs/bench/*.json on the real models (takes a few minutes)
bench:
	cd server && .venv/bin/python scripts/bench_payload.py
	cd server && .venv/bin/python scripts/bench_speed.py
	cd server && .venv/bin/python scripts/bench_scrub.py

# regenerate the four docs/ screenshots (needs smollm2 weights in the hub cache)
shots:
	cd web && node scripts/shots.mjs

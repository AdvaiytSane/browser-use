# Airbnb recording demo

From the repository root, start the loopback dashboard with visible Chromium:

```bash
uv run python examples/integrations/memorable/airbnb_demo_server.py \
  --provider anthropic --env-file .env --no-headless --linger-seconds 5
```

Open `http://127.0.0.1:8766/` and submit a task such as:

```text
Find the least expensive Airbnb in Chicago for two adults from October 16 to October 18, 2026.
```

The dashboard shows the browser snapshot and action stream. Its border color marks deterministic execution, bounded agent selection, popup repair, or safe refusal. Supported result policies are cheapest, highest rated, most reviewed, and first visible. The run is read-only: it does not log in, book, favorite, message, or modify a reservation.

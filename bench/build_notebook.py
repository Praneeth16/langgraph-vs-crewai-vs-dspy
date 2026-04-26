"""Generate notebook.ipynb as an end-to-end walkthrough of the bench.

The notebook is self-contained: clone the repo, install requirements, set
GEMINI_API_KEY, open the notebook, run all. Each cell either imports a
module from `bench/` and runs it, or loads a saved `results/*.json` file
to display previously-computed numbers.

Cells that hit the Gemini API are guarded with a `RUN_LIVE` flag so a
reader can step through with cached results without burning quota.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
nb = {"cells": [], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}


def md(text: str):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text: str):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None})


# ---------------------------------------------------------------------- intro

md("""# LangGraph vs CrewAI vs DSPy: End-to-End Notebook

Companion notebook to `article.md`. Runs the same bench, probes, and compile
experiment the article reports, against three Gemini Flash variants.

**Quick start**

1. `cp .env.example .env` and add your `GEMINI_API_KEY`.
2. `python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. Open this notebook and Run All. Set `RUN_LIVE = False` below to skip live API calls and only display saved results.

The full live re-run costs roughly $5-8 in Gemini API spend and takes about 90 minutes serial. The cached path (RUN_LIVE = False) uses results from `results/*.json` and runs in seconds.
""")

# ---------------------------------------------------------------------- setup

md("## 1. Setup")

code("""import os, sys, json, time
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'bench'))

# Load .env
env_path = ROOT / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

RUN_LIVE = bool(os.environ.get('GEMINI_API_KEY')) and os.environ.get('RUN_LIVE', '0') == '1'
print('GEMINI_API_KEY set:', bool(os.environ.get('GEMINI_API_KEY')))
print('RUN_LIVE:', RUN_LIVE)
""")

# ---------------------------------------------------------------------- dataset

md("""## 2. Dataset

100 unique refund tickets covering 10 issue types (damaged, defective, wrong-item, late, missing-parts, incompatible, quality, change-of-mind, warranty, sizing), 4 length buckets, 5 tones, 5 amount buckets, 3 policy-fit categories. Generator is deterministic (seed=42).
""")

code("""# Show the generator (or regenerate if needed)
ds_path = ROOT / 'bench' / 'tickets_100.json'
if not ds_path.exists():
    !python bench/gen_tickets.py
tickets = json.loads(ds_path.read_text())
print(f'tickets: {len(tickets)}')
print('sample:', tickets[0])
""")

# ---------------------------------------------------------------------- impls

md("## 3. The three implementations")

md("**LangGraph** (state-machine first; checkpoint at every superstep):")
code("(ROOT / 'bench' / 'impl_langgraph.py').read_text()")

md("**CrewAI** (roles inside a Flow; OSS HITL only):")
code("(ROOT / 'bench' / 'impl_crewai.py').read_text()")

md("**DSPy** (Signature + ChainOfThought; cache disabled for fairness):")
code("(ROOT / 'bench' / 'impl_dspy.py').read_text()")

# ---------------------------------------------------------------------- run one

md("""## 4. Run one ticket through each framework

This calls the live Gemini API once per framework if `RUN_LIVE`. Otherwise skipped.
""")

code("""if RUN_LIVE:
    os.environ['GEMINI_MODEL'] = 'gemini-2.5-flash-lite'
    sample = tickets[0]
    for name, mod in [('langgraph', 'impl_langgraph'), ('crewai', 'impl_crewai'), ('dspy', 'impl_dspy')]:
        m = __import__(mod)
        t0 = time.perf_counter()
        out = m.run(sample)
        dt = time.perf_counter() - t0
        print(f'{name}: {dt:.2f}s  draft_reply: {(out.get(\"draft_reply\") or out.get(\"reply\") or \"\")[:120]!r}')
else:
    print('RUN_LIVE=False; skipping single-ticket demo.')
""")

# ---------------------------------------------------------------------- bench

md("""## 5. The 100-ticket bench

The full bench runs 100 tickets across all three frameworks per model, captures latency, prompt/completion tokens, and LLM call count. Live runtime is ~10-12 minutes per model.
""")

code("""if RUN_LIVE:
    # Run all three models
    !cd bench && python bench.py --model gemini-2.5-flash
    !cd bench && python bench.py --model gemini-2.5-flash-lite
    !cd bench && python bench.py --model gemini-3.1-flash-lite-preview
""")

md("**Cached results: latency table with 95% CIs (computed by `enrich.py`).**")

code("""!python bench/enrich.py 2>/dev/null | head -30""")

# ---------------------------------------------------------------------- probes

md("""## 6. Probes

Four probes pull additional structure out of the bench:

- `probe_overhead.py` runs a single fixed prompt through each framework to isolate template overhead.
- `inspect_messages.py` captures the literal byte-level prompt each framework sends.
- `probe_reasoning.py` decomposes completion tokens into reasoning vs non-reasoning per framework on Gemini 2.5 Flash.
- `probe_cprofile.py` wraps a single-ticket run in cProfile per framework and captures function call counts.
""")

code("""if RUN_LIVE:
    !python bench/probe_overhead.py
    !python bench/inspect_messages.py
    !python bench/probe_reasoning.py
    !python bench/probe_cprofile.py
""")

md("**Cached: framework prompt overhead (probe_overhead.json):**")
code("""p = ROOT / 'results' / 'probe_overhead.json'
if p.exists():
    d = json.loads(p.read_text())
    for model, rows in d.items():
        print(f'\\n=== {model} ===')
        for r in rows:
            print(f'  {r}')
""")

md("**Cached: byte-level prompt each framework sent (messages_inspected.json):**")
code("""p = ROOT / 'results' / 'messages_inspected.json'
if p.exists():
    d = json.loads(p.read_text())
    for fw, msgs in d.items():
        print(f'\\n=== {fw} ===')
        for m in msgs:
            print(f'[{m[\"role\"]}]')
            print(m['content'][:600])
""")

md("**Cached: reasoning fraction with bootstrap CIs (probe_reasoning.json):**")
code("""import statistics, random, math
p = ROOT / 'results' / 'probe_reasoning.json'
if p.exists():
    d = json.loads(p.read_text())
    def boot(xs, B=2000, seed=42):
        rng = random.Random(seed); n = len(xs); means = []
        for _ in range(B):
            s = [xs[rng.randrange(n)] for _ in range(n)]
            means.append(sum(s) / n)
        means.sort()
        return statistics.mean(means), means[int(0.025*B)], means[int(0.975*B)]
    print(f'{\"framework\":<10} {\"mean reason%\":>12} {\"95% CI\":>20} {\"n\":>4}')
    for fw, rows in d.items():
        fracs = [r['reasoning']/r['completion'] for r in rows if 'error' not in r and r['completion']]
        m, lo, hi = boot(fracs)
        print(f'{fw:<10} {m*100:>11.1f}% {f\"[{lo*100:.1f}%, {hi*100:.1f}%]\":>20} {len(fracs):>4}')
""")

md("**Cached: cProfile function-call counts (probe_cprofile.json):**")
code("""p = ROOT / 'results' / 'probe_cprofile.json'
if p.exists():
    d = json.loads(p.read_text())
    print(f'{\"framework\":<10} {\"wall_s\":>8} {\"function_calls (from pstats)\":>40}')
    for fw, info in d.items():
        head = info['full_pstats'].splitlines()[0] if info.get('full_pstats') else ''
        print(f'{fw:<10} {info.get(\"wall_s\", 0):>8} {head}')
""")

# ---------------------------------------------------------------------- compile

md("""## 7. The DSPy compile question

Two experiments. `compile_dspy.py` runs BootstrapFewShot at five seeds (42-45 plus a no-shuffle baseline). `hand_fewshot_lg.py` runs three hand-picked 4-shot LangGraph variants on the same test set. Comparison answers: is the compile lift attributable to compilation or just to having 4 examples in context?
""")

code("""if RUN_LIVE:
    os.environ['GEMINI_MODEL'] = 'gemini-2.5-flash-lite'
    for s in [42, 43, 44, 45]:
        !cd bench && GEMINI_MODEL=gemini-2.5-flash-lite python compile_dspy.py --seed {s}
    for v in ['v1', 'v2', 'v3']:
        !cd bench && GEMINI_MODEL=gemini-2.5-flash-lite python hand_fewshot_lg.py --variant {v}
""")

md("**Cached: compile + hand-pick comparison.**")

code("""rows = []
# Compiled DSPy seeds
for s in [42, 43, 44, 45]:
    p = ROOT / 'results' / f'dspy_compiled_eval_gemini-2.5-flash-lite_seed{s}.json'
    if p.exists():
        d = json.loads(p.read_text())
        ok = [r for r in d['compiled']['rows'] if 'error' not in r]
        n = len(ok)
        acc = sum(r['correct'] for r in ok) / n
        toks = sum(r['total'] for r in ok) / n
        rows.append(('compiled', f'seed{s}', acc, toks, n))
# Hand-picks
for v in ['v1', 'v2', 'v3']:
    p = ROOT / 'results' / f'hand_fewshot_lg_eval_gemini-2.5-flash-lite_{v}.json'
    if p.exists():
        d = json.loads(p.read_text())
        ok = [r for r in d['rows'] if 'error' not in r]
        n = len(ok)
        acc = sum(r['correct'] for r in ok) / n
        toks = sum(r['prompt']+r['completion'] for r in ok) / n
        rows.append(('hand-4shot', v, acc, toks, n))

print(f'{\"variant\":<12} {\"id\":>8} {\"acc\":>6} {\"tokens/ticket\":>15} {\"n\":>4}')
for k, vid, acc, toks, n in rows:
    print(f'{k:<12} {vid:>8} {acc:>5.1%} {toks:>14.0f} {n:>4}')
""")

# ---------------------------------------------------------------------- closer

md("""## 8. Where to go next

- Read `article.md` for the full narrative, with the verdict at the end.
- Read `results/analysis_v2.md` for audited stat tables.
- Re-run any cell with `RUN_LIVE=1 GEMINI_API_KEY=... jupyter nbconvert --execute ...` against your own model setup.
- Re-run on your workload by replacing `bench/tickets_100.json` with your own ticket-shaped data; the rest of the bench is workload-agnostic.

The bench code is the artifact. The numbers shipped here are one snapshot.
""")

# ---------------------------------------------------------------------- write

# Wrap source strings into list-of-strings format Jupyter expects when written
for cell in nb["cells"]:
    if isinstance(cell["source"], str):
        cell["source"] = cell["source"].splitlines(keepends=True)

out = ROOT / "notebook.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out}")

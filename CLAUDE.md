# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo Layout

Writing project with a runnable benchmark harness. Three-way comparison of LangGraph vs CrewAI vs DSPy for production LM systems.

- `article.md`: the article (sole prose artifact).
- `01_*.png` through `04_*.png`: four inline figures.
- `bench/`: runnable benchmark.
  - `impl_langgraph.py`, `impl_crewai.py`, `impl_dspy.py`: same refund-assistant use case, three frameworks.
  - `bench.py`: harness. 100 tickets, multi-model via `--model`, emits `results/bench_<model>.json`. Richer metrics (p50/p90/p95/p99, prompt+completion breakdown, per-ticket meta).
  - `gen_tickets.py`: generator for `tickets_100.json` (10 issue types x 4 lengths x 5 tones x 5 amount buckets x 3 policy-fit categories).
  - `tickets_100.json`: 100 unique refund tickets with metadata.
  - `probe_overhead.py`: fixed-prompt microbench isolating framework overhead on a single LLM call.
  - `analyze.py`: post-hoc analysis producing `results/analysis.md` with tail latency, per-issue-type, length correlation, cross-framework agreement, outliers.
  - `enrich.py`: stat-rigor pass producing `results/analysis_v2.md` (95% CIs on means, bootstrap CIs on p95/p99, Spearman + Pearson correlations, $/ticket cost, audited cross-framework agreement).
  - `probe_reasoning.py`: per-framework `output_token_details.reasoning` decomposition on 2.5 Flash. Output `results/probe_reasoning.json`.
  - `probe_cprofile.py`: cProfile per framework on a single ticket. Output `results/probe_cprofile.json`.
  - `inspect_messages.py`: captures byte-level prompts each framework sends. Output `results/messages_inspected.json`.
  - `compile_dspy.py`: BootstrapFewShot compile of the DSPy refund program; uncompiled vs compiled accuracy + tokens. Accepts `--seed`. Output `results/dspy_compiled_eval_<model>_seed<N>.json`.
  - `hand_fewshot_lg.py`: hand-written 4-shot LangGraph baseline on the same 50-ticket test set as `compile_dspy.py`. Three demo-set variants (`--variant v1|v2|v3`) for selection-bias control. Output `results/hand_fewshot_lg_eval_<model>_<variant>.json`.
  - `build_notebook.py`: rebuilds `notebook.ipynb` from the impl files plus latest results.
- `notebook.ipynb`: self-contained walk-through.
- `.env`: gitignored. Holds `GEMINI_API_KEY`.
- `.venv/`: Python 3.13 venv. Activate with `source .venv/bin/activate`.
- `results/bench_gemini-2.5-flash.json`, `results/bench_gemini-2.5-flash-lite.json`, `results/bench_gemini-3.1-flash-lite-preview.json`, `results/analysis.md`, `results/analysis_v2.md`, `results/probe_overhead.json`, `results/probe_reasoning.json`, `results/probe_cprofile.json`, `results/messages_inspected.json`, `results/dspy_compiled_eval_gemini-2.5-flash-lite.json`: authoritative measurement outputs.

## Running the Bench

```bash
set -a && . ./.env && set +a
source .venv/bin/activate
cd bench && python bench.py --model gemini-2.5-flash          # writes results/bench_gemini-2.5-flash.json
cd bench && python bench.py --model gemini-3.1-flash-lite-preview
python probe_overhead.py                                      # writes results/probe_overhead.json
python analyze.py                                             # writes results/analysis.md
python build_notebook.py                                      # rebuilds notebook.ipynb from impls + results
```

Full bench (100 tickets, all three frameworks, one model) takes roughly 12 minutes on Gemini 2.5 Flash and roughly 10 minutes on 3.1 Flash Lite.

## CrewAI HITL Gotcha

`@human_feedback` from `crewai.flow.human_feedback` is **CrewAI Enterprise only**. The OSS pip package (tested at 0.134.0) does not expose this module. An older revision of this article had code that would not run. OSS HITL mechanisms: `Task(human_input=True)` at the task level, or explicit state-level branches inside a Flow listener (see `impl_crewai.py`). Do not reintroduce the enterprise import path in example code without flagging it.

## Figures Are Stale Relative to Current Article

The four PNGs were produced when the article was a two-way LangGraph vs CrewAI compare. The article is now three-way. Each figure caption in `article.md` explicitly acknowledges the gap (e.g. "Figure 1 shows the CrewAI vs LangGraph polarity; DSPy sits on a third axis"). If you regenerate figures, drop the acknowledgment language from the captions. If you rename or reorder sections, re-check that each `![...](0X_*.png)` still lands in the section its caption describes.

## Editing Conventions

- **No em-dashes and no en-dashes anywhere in prose, code, captions, or CLAUDE.md.** User rule. Use periods, colons, commas, or parentheses instead.
- **Thesis is triangular, not linear.** CrewAI optimizes collaboration (roles). LangGraph optimizes state (runtime). DSPy optimizes the program itself (compiled prompts and weights against a metric). The three are not on the same axis; the composition section argues DSPy typically stacks inside LangGraph nodes or CrewAI tasks. Do not flatten back to a linear ranking.
- **Measurements are grounded.** Tables come from 100-ticket runs (`bench/tickets_100.json`) against three models, plus reasoning-token, cProfile, and compiled-DSPy probes. Means with 95% CI: 2.5 Flash LG 6.99 [6.37, 7.60] / CA 7.28 [6.13, 8.44] / DSPy 4.37 [3.98, 4.77]. 2.5-Lite LG 2.12 [2.05, 2.18] / CA 2.71 [2.61, 2.81] / DSPy 1.83 [1.67, 1.99]. 3.1-Lite Preview LG 2.16 [2.04, 2.28] / CA 2.27 [2.17, 2.38] / DSPy 1.71 [1.56, 1.86]. (Non-Lite `gemini-3.1-flash` does not exist in the public API as of 2026-04.) p99 with bootstrap 95% CI: CrewAI on 2.5 Flash 41.24 [14.50, 43.08] *overlaps* LangGraph 19.11 [14.03, 21.82] - tail-difference claims must be hedged. Reasoning fraction of completion on 2.5 Flash: LG 92.5%, CA 91.5%, DSPy 69.0%. cProfile function calls per ticket: LG 80k, CA 86k, DSPy 384k. Compiled DSPy on 2.5-Lite: 68% -> 82% acc, 2.1x token cost, 6.2s compile wall. $/1k tickets: 2.5 Flash $1.33-$2.53; Lite tier $0.05-$0.14. Cold import 0.35s/2.00s/0.08s and LoC 85/89/62 unchanged. Sources: bench_*.json, analysis_v2.md, probe_reasoning.json, probe_cprofile.json, messages_inspected.json, dspy_compiled_eval_*.json. Do not hand-edit; re-run the bench.
- **DSPy cache must be off for fair comparison.** `dspy.LM(..., cache=True)` is the default. Cached runs return in ~0.3s total with zero recorded tokens. `impl_dspy.py` sets `cache=False`. Do not remove that without noting it in the article.
- **Library versions pinned to the code trace.** The "Why the Overhead" section of the article quotes file:line from `langgraph==1.1.6`, `crewai==0.134.0`, `dspy==3.1.3`, `litellm==1.72.0`. Bumping any of these can invalidate cited line numbers; re-verify before releasing.
- **Named deployment claims trace to References.** Klarna 85M / ~80% resolution-time cut, Uber Lang Effect / 5k engineers / ~21k dev-hours / 100 parallel test iterations, AppFolio 10+ hrs/week and doubled decision accuracy, LinkedIn, Replit, Elastic for LangGraph. PwC/IBM/Capgemini/NVIDIA and the 2B-execution figure for CrewAI (flagged as self-reported). JetBlue/Replit/Moody's/Databricks/JPMC/VMware/Sephora/Normal Computing/Haize for DSPy. MIPROv2 and GEPA (arXiv 2507.19457, ICLR 2026 Oral), GEPA beating MIPROv2 by ~10 points and GRPO by up to 20 points with up to 35x fewer rollouts. Do not add new named claims without adding a reference.
- **Asymmetry is load-bearing.** CrewAI checkpointing is early-release. LangGraph checkpointing is spine-of-the-system. DSPy has no runtime checkpointing concept at all (different category). Keep this sharp; it is the reason the composition section exists.
- **No repetition.** Current structure: How to Read This (no top TL;DR), Three Centers of Gravity, Real Production Footprint, Where Each One Breaks, One Product Three Code Paths, What the Measurements Say, Why the Overhead Exists (code trace + cProfile + reasoning decomposition), Compiled DSPy (multi-seed + multi-variant hand 4-shot baseline), Post-Hoc (5 obs + 3 appendix), Composition, Decision Rules (anti-recs + traps only), Limitations, Reproduction Recipe, TL;DR / Final Take (combined verdict at end), References. The verdict lives at the end. Do not reintroduce a top-of-article TL;DR.
- **No first-person changelog voice in the article.** Phrases like "I previously wrote" or "I want to soften this claim" belong in CLAUDE.md or a retro, not the artifact. State the current claim directly.
- **Compiled-DSPy result is multi-seed (5 runs) and compared against multi-variant hand 4-shot LangGraph baseline (3 demo sets).** Compile: range 70-84%, mean 75.2%, stdev 7.2pp; only 2 of 5 seeds reach McNemar's α=0.05 over uncompiled. Hand-4-shot: range 70-76%, mean 74.0%, stdev ~3.5pp at 310-339 tokens/ticket vs compile mean 754 tokens/ticket. Conclusion: compile lift on this trainset/optimizer is mostly few-shot prompting; demo curation matters at least as much as the compile algorithm. Do not report only the favorable seed.
- **Reasoning-fraction probe uses bootstrap CI on per-ticket fractions (not Wilson on calls).** LangGraph 90.7% [89.5, 91.9], CrewAI 89.2% [87.9, 90.4], DSPy 64.1% [60.1, 68.4] on 2.5 Flash. DSPy CI cleanly separated from the other two; LG vs CA is not.

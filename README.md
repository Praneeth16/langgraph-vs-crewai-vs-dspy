# LangGraph vs CrewAI vs DSPy

A 900-run benchmark and a long-form article comparing three production-grade frameworks for LLM applications. The bench answers questions the marketing pages don't: how much does each framework actually add to your prompt, where does the model spend its reasoning budget, and is "compile your DSPy program" really worth the token cost.

> **TL;DR**
> Tier choice (Gemini 2.5 Flash vs Flash-Lite) cuts cost 25-50x on this workload. Framework choice is a 1.7-2.7x optimization on top. CrewAI vs LangGraph mean-latency CIs overlap on Flash. DSPy is consistently the lowest-latency framework, partly through reasoning-token suppression and partly through short-circuit logic. BootstrapFewShot compile across 5 seeds matches a hand-written 4-shot LangGraph baseline at half the token cost on a 50-ticket eval. **Tier choice compounds faster than framework choice. Try Flash-Lite first.**

Read the full article: [article.md](./article.md).

Walk the bench end-to-end: [notebook.ipynb](./notebook.ipynb).

## What's in the box

- **Three framework implementations of the same refund-assistant workflow** (`bench/impl_langgraph.py`, `bench/impl_crewai.py`, `bench/impl_dspy.py`). Each reads a ticket, decides policy fit, drafts a reply, pauses for approval above a threshold, sends. About 60-90 LoC each.
- **A 100-ticket synthetic dataset** (`bench/tickets_100.json`) covering 10 issue types, 5 customer tones, 4 length buckets, 5 amount buckets, 3 policy-fit categories. Generator (`bench/gen_tickets.py`) is deterministic at `seed=42`.
- **A bench harness** (`bench/bench.py`) that runs all three frameworks on the dataset against a model you choose, captures latency, prompt and completion tokens, LLM call counts, and per-ticket meta.
- **Probes for what aggregate numbers hide**:
  - `bench/probe_overhead.py` runs one fixed prompt through each framework and reports prompt-token delta vs raw HTTP.
  - `bench/inspect_messages.py` captures the literal byte-level prompt each framework sends, tokenized via Gemini's `countTokens`.
  - `bench/probe_reasoning.py` decomposes completion tokens into reasoning vs non-reasoning per framework on Gemini 2.5 Flash.
  - `bench/probe_cprofile.py` cProfiles a single-ticket run per framework and captures function-call counts.
- **The DSPy compile question, answered with controls**:
  - `bench/compile_dspy.py` runs `BootstrapFewShot` across multiple seeds (`--seed 42|43|44|45`) on a 50-train / 50-eval split with `gemini-2.5-flash-lite`.
  - `bench/hand_fewshot_lg.py` runs three hand-picked 4-shot LangGraph baselines (`--variant v1|v2|v3`) on the same test set, controlling whether observed lift is attributable to "DSPy compilation" or "any 4 examples in the prompt".
- **Audited stat tables** (`bench/enrich.py` -> `results/analysis_v2.md`): mean 95% CIs, bootstrap 95% CIs on p95/p99, Pearson + Spearman correlations, $/1k-tickets cost at Gemini list pricing, audited cross-framework decision agreement.

## Quick start

```bash
git clone https://github.com/Praneeth16/langgraph-vs-crewai-vs-dspy.git
cd langgraph-vs-crewai-vs-dspy
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env to set GEMINI_API_KEY=<your-key>
```

Then either open the notebook (`jupyter notebook notebook.ipynb` and Run All) or run the bench from the command line:

```bash
set -a && . ./.env && set +a
cd bench
python bench.py --model gemini-2.5-flash-lite
python probe_reasoning.py
python compile_dspy.py --seed 42
python hand_fewshot_lg.py --variant v1
cd ..
python bench/enrich.py
```

A full live re-run (3 models x 100 tickets x 3 frameworks + probes + multi-seed compile + multi-variant hand-pick) is roughly 90 minutes wall time and $5-8 in Gemini API spend at list pricing.

The notebook supports `RUN_LIVE=False` (the default when no API key is set) to walk through using cached `results/*.json` files without burning quota.

## Headline numbers

| Model | LangGraph | CrewAI | DSPy |
|---|---|---|---|
| Gemini 2.5 Flash | 6.99 [6.37, 7.60] | 7.28 [6.13, 8.44] | 4.37 [3.98, 4.77] |
| Gemini 2.5 Flash-Lite | 2.12 [2.05, 2.18] | 2.71 [2.61, 2.81] | 1.83 [1.67, 1.99] |
| Gemini 3.1 Flash-Lite Preview | 2.16 [2.04, 2.28] | 2.27 [2.17, 2.38] | 1.71 [1.56, 1.86] |

*Mean latency in seconds with 95% t-CI, n=100 per cell.*

| Model | LangGraph | CrewAI | DSPy |
|---|---|---|---|
| 2.5 Flash | $2.27 | $2.53 | $1.33 |
| 2.5 Flash-Lite | $0.05 | $0.14 | $0.10 |
| 3.1 Flash-Lite Preview | $0.05 | $0.10 | $0.10 |

*Cost per 1,000 tickets at Gemini API list pricing. 3.1 Flash-Lite Preview pricing is approximated using 2.5-Lite as proxy.*

Every other table (token decomposition, cProfile counts, reasoning fractions with bootstrap CIs, multi-seed compile vs hand-pick comparison, confusion matrices) sits in `article.md`.

## Repo layout

```
.
+-- article.md                 The article. Read this end-to-end.
+-- notebook.ipynb             End-to-end walkthrough; runs the bench or shows cached results.
+-- README.md                  This file.
+-- requirements.txt           Pinned deps (langgraph 1.1.6, crewai 0.134.0, dspy 3.1.3, litellm 1.72.0).
+-- .env.example               Copy to .env and add GEMINI_API_KEY.
+-- 01_*.png ... 04_*.png      Inline figures referenced from article.md.
+-- bench/
|   +-- bench.py               Main 100-ticket harness (--model, --frameworks, --limit, --out).
|   +-- gen_tickets.py         Deterministic dataset generator.
|   +-- tickets_100.json       The dataset.
|   +-- impl_langgraph.py      LangGraph refund flow.
|   +-- impl_crewai.py         CrewAI refund flow.
|   +-- impl_dspy.py           DSPy refund program (cache=False for fairness).
|   +-- probe_overhead.py      Single fixed-prompt overhead per framework.
|   +-- probe_reasoning.py     Reasoning-token decomposition.
|   +-- probe_cprofile.py      cProfile per framework.
|   +-- inspect_messages.py    Captures byte-level prompt sent by each framework.
|   +-- compile_dspy.py        BootstrapFewShot multi-seed compile + paired McNemar's.
|   +-- hand_fewshot_lg.py     Hand-picked 4-shot LangGraph baseline; --variant v1|v2|v3.
|   +-- analyze.py             Per-issue-type, outliers, errors -> results/analysis.md.
|   +-- enrich.py              CIs, bootstrap, Spearman, cost -> results/analysis_v2.md.
|   +-- build_notebook.py      Regenerates notebook.ipynb.
+-- results/                   Bench outputs (gitignored *.json; analysis_v2.md committed).
```

## How the bench answers the questions

- **"Is CrewAI's tail really worse than LangGraph's?"** -> Bootstrap CIs on p99 overlap on 2.5 Flash. The difference is variance, not the average. See "Five Stories the Bench Tells" in the article.
- **"Where does CrewAI's extra prompt cost come from?"** -> The OSS package emits a hardcoded ReAct envelope (~96 system tokens + ~94 user tokens) on every LLM call regardless of `verbose` flag. The literal prompt is captured in `results/messages_inspected.json`. Article section: "Where the Tokens Actually Go".
- **"Does DSPy compile actually help on small classification?"** -> Across 5 BootstrapFewShot seeds, accuracy 70-84%. A hand-written 4-shot LangGraph baseline across 3 demo sets lands at 70-76%. Welch's t on the means does not reject the null. Article section: "The DSPy Compile Question".
- **"Why is DSPy faster on Flash?"** -> Two reasons. The ChatAdapter format suppresses Gemini 2.5 Flash's reasoning emission to 64.1% of completion (CI [60.1%, 68.4%]) vs ~90% for LG and CrewAI. And the program short-circuits on ineligible triage, lowering calls/ticket from 2.0 to 1.18-1.25. The first effect is structural to DSPy. The second is portable.
- **"Will my migration from CrewAI to LangGraph 'just work'?"** -> Same 100 tickets at temperature zero produce identical decisions across all three frameworks only 15-20% of the time. The runtime is not the source of variance; the prompt template is. Build a regression eval first.

## Library versions

The article quotes file:line citations against pinned versions. Bumping them may invalidate cited line numbers; verify before treating quoted code as canonical.

```
langgraph==1.1.6
crewai==0.134.0
dspy==3.1.3
litellm==1.72.0
langchain-google-genai==4.2.2
langchain-core==1.3.0
```

## Known limitations

Spelled out at length in the article's "Where the Findings Stop" section. Headline ones:

- One workload (refund-eligibility classification with 2 LLM calls per ticket).
- Three model variants, all from the Gemini Flash family.
- Compile evaluation is small (n=50 test); Wilson 95% CIs are ~12 percentage points wide.
- Only one optimizer (BootstrapFewShot at default settings) tested. MIPROv2 and GEPA at larger trainsets likely produce a more reliable lift.
- Reasoning probe is n=30 tickets per framework.
- `gemini-3.1-flash-lite-preview` is a preview model; final pricing has not been announced.

## Reproduction

The notebook is the friendliest path. Command-line reproduction is in `article.md` -> "Reproduction Recipe". The dataset is deterministic; the bench captures latency from a single laptop run, so absolute wall-times will differ on your machine, but token counts, reasoning fractions, and per-framework comparisons should reproduce.

## License

MIT.

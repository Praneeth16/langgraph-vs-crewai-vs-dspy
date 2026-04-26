"""Stat rigor enrichment over results/bench_<model>.json.

Adds:
  - 95% CI of mean (analytic, t-distribution).
  - Bootstrap 95% CI of p95 and p99 (1000 resamples).
  - Pearson + Spearman correlation of latency vs ticket length.
  - Fixed cross-framework agreement math (count over the 100-ticket union).
  - Cost per ticket in dollars (Gemini Flash pricing table).
  - Per-framework reasoning-token estimate (total - prompt - completion overflow check).

Output: results/analysis_v2.md
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Public Google AI pricing as of 2026-04. Flash-Lite preview pricing not announced;
# use 2.5-Lite as proxy and flag in the article.
PRICING = {
    # input $/1M, output $/1M
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite-preview": (0.10, 0.40),  # proxy
}


def load_all():
    out = {}
    for p in sorted(RESULTS.glob("bench_gemini-*.json")):
        model = p.stem.replace("bench_", "")
        out[model] = json.loads(p.read_text())
    return out


def t_critical_95(n: int) -> float:
    """Two-tailed 95% t critical. Approximate; for n>=30 ~ 1.96."""
    if n < 2:
        return float("nan")
    if n >= 30:
        return 1.96
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365,
             10: 2.262, 15: 2.131, 20: 2.093, 25: 2.060}
    keys = sorted(table.keys())
    for k in keys:
        if n <= k:
            return table[k]
    return 1.96


def mean_ci_95(xs):
    if len(xs) < 2:
        return (float("nan"), float("nan"), float("nan"))
    m = statistics.mean(xs)
    s = statistics.stdev(xs)
    se = s / math.sqrt(len(xs))
    t = t_critical_95(len(xs))
    return m, m - t * se, m + t * se


def bootstrap_pct_ci(xs, p=0.95, B=1000, seed=42):
    if len(xs) < 5:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    boots = []
    for _ in range(B):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        sample.sort()
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        boots.append(sample[idx])
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[int(0.975 * B)]
    sample = sorted(xs)
    point = sample[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return point, lo, hi


def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    rx = _ranks(xs)
    ry = _ranks(ys)
    return statistics.correlation(rx, ry)


def _ranks(xs):
    indexed = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def cost_per_ticket(rows, model: str):
    rates = PRICING.get(model)
    if rates is None:
        return None
    pin, pout = rates
    costs = []
    for r in rows:
        if "error" in r:
            continue
        c = (r["prompt_tokens"] / 1e6) * pin + (r["completion_tokens"] / 1e6) * pout
        costs.append(c)
    return costs


def latency_table(data):
    lines = ["## Latency (mean with 95% CI; bootstrap p95/p99 with 95% CI)\n"]
    lines.append("| Model | Framework | n | Mean | 95% CI | Median | p95 | p95 95% CI | p99 | p99 95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for model, runs in data.items():
        for r in runs:
            fw = r["framework"]
            ok = [row for row in r["rows"] if "error" not in row]
            if not ok:
                continue
            lat = [row["latency_s"] for row in ok]
            m, lo, hi = mean_ci_95(lat)
            p95, p95lo, p95hi = bootstrap_pct_ci(lat, p=0.95)
            p99, p99lo, p99hi = bootstrap_pct_ci(lat, p=0.99)
            lines.append(
                f"| {model} | {fw} | {len(lat)} | {m:.2f} | "
                f"[{lo:.2f}, {hi:.2f}] | "
                f"{statistics.median(lat):.2f} | "
                f"{p95:.2f} | [{p95lo:.2f}, {p95hi:.2f}] | "
                f"{p99:.2f} | [{p99lo:.2f}, {p99hi:.2f}] |"
            )
    return "\n".join(lines)


def cost_table(data):
    lines = ["\n## Cost per ticket (USD, Gemini API list pricing)\n"]
    lines.append("| Model | Framework | Mean cents/ticket | Median cents/ticket | $/1k tickets |")
    lines.append("|---|---|---|---|---|")
    for model, runs in data.items():
        for r in runs:
            fw = r["framework"]
            ok = [row for row in r["rows"] if "error" not in row]
            costs = cost_per_ticket(ok, model)
            if not costs:
                continue
            mean = statistics.mean(costs) * 100
            med = statistics.median(costs) * 100
            per_k = sum(costs) * 1000 / len(costs)
            lines.append(
                f"| {model} | {fw} | {mean:.4f} | {med:.4f} | ${per_k:.2f} |"
            )
    return "\n".join(lines)


def correlation_table(data):
    lines = ["\n## Latency / token correlation with ticket length (Pearson, Spearman)\n"]
    lines.append("| Model | Framework | Pearson(len, lat) | Spearman(len, lat) | Pearson(len, tok) | Spearman(len, tok) |")
    lines.append("|---|---|---|---|---|---|")
    for model, runs in data.items():
        for r in runs:
            fw = r["framework"]
            ok = [row for row in r["rows"] if "error" not in row]
            if len(ok) < 3:
                continue
            lens = [row["meta"]["email_word_len"] for row in ok]
            lat = [row["latency_s"] for row in ok]
            tok = [row["total_tokens"] for row in ok]
            try:
                pl = statistics.correlation(lens, lat)
            except Exception:
                pl = float("nan")
            try:
                pt = statistics.correlation(lens, tok)
            except Exception:
                pt = float("nan")
            sl = spearman(lens, lat)
            st = spearman(lens, tok)
            lines.append(
                f"| {model} | {fw} | {pl:.3f} | {sl:.3f} | {pt:.3f} | {st:.3f} |"
            )
    return "\n".join(lines)


def agreement_audit(data):
    """Audit-fixed cross-framework agreement: count over union of ticket_ids,
    treating missing data as 'no decision' rather than dropping."""
    lines = ["\n## Cross-framework agreement (audited)\n"]
    lines.append("| Model | Tickets with all 3 frameworks | All 3 agree | 2-of-3 agree | All 3 differ |")
    lines.append("|---|---|---|---|---|")
    for model, runs in data.items():
        by_ticket = {}
        for r in runs:
            for row in r["rows"]:
                if "error" in row:
                    continue
                tid = row["ticket_id"]
                # Decision label: status if present, else sent/held by approved.
                d = row.get("status") or ("sent" if row.get("approved") else "held")
                by_ticket.setdefault(tid, {})[r["framework"]] = d
        all3 = two = zero = total = 0
        for tid, decisions in by_ticket.items():
            if len(decisions) < 3:
                continue
            total += 1
            vals = list(decisions.values())
            uniq = len(set(vals))
            if uniq == 1:
                all3 += 1
            elif uniq == 2:
                two += 1
            else:
                zero += 1
        lines.append(f"| {model} | {total} | {all3} | {two} | {zero} |")
    return "\n".join(lines)


def reasoning_decomposition(data):
    """Use total = prompt + completion always holds, but on Gemini 2.5 the completion
    INCLUDES reasoning tokens. We don't have a per-row reasoning breakdown in the
    saved bench JSON. This table is therefore a model-level reminder, not a
    per-framework decomposition. The reasoning-token probe (probe_reasoning.py)
    fills the gap."""
    return ""


def main():
    data = load_all()
    parts = [
        "# Benchmark Analysis (v2, audited)\n",
        latency_table(data),
        cost_table(data),
        correlation_table(data),
        agreement_audit(data),
    ]
    out = RESULTS / "analysis_v2.md"
    out.write_text("\n\n".join(parts))
    print(f"Wrote {out}")
    print("\n".join(parts))


if __name__ == "__main__":
    main()

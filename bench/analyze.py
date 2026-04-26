"""Post-hoc analysis across frameworks and models.

Reads results/bench_<model>.json files, produces:
  - latency distribution tables (mean, median, p50, p90, p95, p99, stdev)
  - token breakdown (prompt vs completion) per framework x model
  - per-issue-type latency to show where each framework struggles
  - ticket-length vs latency correlation
  - cross-framework agreement on eligibility decisions (reliability proxy)
  - outlier list (slowest 5 per framework x model)

Output: results/analysis.md + results/analysis.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_all():
    out = {}
    for p in sorted(RESULTS.glob("bench_*.json")):
        model = p.stem.replace("bench_", "")
        out[model] = json.loads(p.read_text())
    return out


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return round(s[idx], 3)


def latency_table(data):
    lines = ["## Latency distribution (seconds)\n"]
    lines.append("| Model | Framework | n | mean | median | p50 | p90 | p95 | p99 | stdev | max |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, runs in data.items():
        for r in runs:
            fw = r["framework"]
            ok = [row for row in r["rows"] if "error" not in row]
            lat = [row["latency_s"] for row in ok]
            if not lat:
                continue
            lines.append(
                f"| {model} | {fw} | {len(lat)} | "
                f"{round(statistics.mean(lat), 2)} | "
                f"{round(statistics.median(lat), 2)} | "
                f"{pct(lat, 0.50)} | {pct(lat, 0.90)} | "
                f"{pct(lat, 0.95)} | {pct(lat, 0.99)} | "
                f"{round(statistics.pstdev(lat), 2)} | "
                f"{round(max(lat), 2)} |"
            )
    return "\n".join(lines)


def token_table(data):
    lines = ["\n## Token breakdown per ticket\n"]
    lines.append("| Model | Framework | Mean prompt | Mean completion | Mean total | Median total | stdev | Ratio completion/prompt |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for model, runs in data.items():
        for r in runs:
            fw = r["framework"]
            ok = [row for row in r["rows"] if "error" not in row]
            if not ok:
                continue
            prompt = [row["prompt_tokens"] for row in ok]
            comp = [row["completion_tokens"] for row in ok]
            total = [row["total_tokens"] for row in ok]
            ratio = round(sum(comp) / max(sum(prompt), 1), 2)
            lines.append(
                f"| {model} | {fw} | "
                f"{round(statistics.mean(prompt))} | "
                f"{round(statistics.mean(comp))} | "
                f"{round(statistics.mean(total))} | "
                f"{round(statistics.median(total))} | "
                f"{round(statistics.pstdev(total))} | {ratio} |"
            )
    return "\n".join(lines)


def by_issue(data):
    lines = ["\n## Mean latency by issue type (seconds)\n"]
    rows = {}
    issues = set()
    for model, runs in data.items():
        for r in runs:
            fw = r["framework"]
            key = (model, fw)
            rows.setdefault(key, {})
            for row in r["rows"]:
                if "error" in row:
                    continue
                issue = row.get("meta", {}).get("issue", "?")
                issues.add(issue)
                rows[key].setdefault(issue, []).append(row["latency_s"])

    issues = sorted(issues)
    header = "| Model | Framework | " + " | ".join(issues) + " |"
    sep = "|" + "---|" * (2 + len(issues))
    lines.append(header)
    lines.append(sep)
    for (model, fw), by_iss in rows.items():
        vals = [round(statistics.mean(by_iss[i]), 2) if by_iss.get(i) else "-" for i in issues]
        lines.append(f"| {model} | {fw} | " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines)


def length_correlation(data):
    lines = ["\n## Latency scaling with ticket length\n"]
    lines.append("| Model | Framework | Corr(length, latency) | Corr(length, tokens) |")
    lines.append("|---|---|---|---|")
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
                cl = round(statistics.correlation(lens, lat), 3)
            except statistics.StatisticsError:
                cl = None
            try:
                ct = round(statistics.correlation(lens, tok), 3)
            except statistics.StatisticsError:
                ct = None
            lines.append(f"| {model} | {fw} | {cl} | {ct} |")
    return "\n".join(lines)


def outliers(data, k=5):
    lines = ["\n## Slowest tickets per framework x model\n"]
    for model, runs in data.items():
        for r in runs:
            fw = r["framework"]
            ok = [row for row in r["rows"] if "error" not in row]
            if not ok:
                continue
            worst = sorted(ok, key=lambda x: x["latency_s"], reverse=True)[:k]
            lines.append(f"\n**{model} {fw}**: top-{k} slowest")
            for w in worst:
                lines.append(
                    f"- {w['ticket_id']} ({w['meta']['issue']}, "
                    f"{w['meta']['email_word_len']} words): "
                    f"{w['latency_s']}s, {w['total_tokens']} tokens"
                )
    return "\n".join(lines)


def errors_summary(data):
    lines = ["\n## Errors and timeouts\n"]
    lines.append("| Model | Framework | Total rows | OK | Timed out | Other errors |")
    lines.append("|---|---|---|---|---|---|")
    for model, runs in data.items():
        for r in runs:
            fw = r["framework"]
            rows = r["rows"]
            ok = [row for row in rows if "error" not in row]
            timed = [row for row in rows if row.get("timed_out")]
            other = [row for row in rows if "error" in row and not row.get("timed_out")]
            lines.append(f"| {model} | {fw} | {len(rows)} | {len(ok)} | {len(timed)} | {len(other)} |")
    return "\n".join(lines)


def agreement(data):
    """For each model, how often do all 3 frameworks agree on approved/held?"""
    lines = ["\n## Cross-framework agreement on decision (per model)\n"]
    lines.append("| Model | All 3 agree | 2-of-3 agree | All 3 differ |")
    lines.append("|---|---|---|---|")
    for model, runs in data.items():
        by_ticket = {}
        for r in runs:
            for row in r["rows"]:
                if "error" in row:
                    continue
                by_ticket.setdefault(row["ticket_id"], {})[r["framework"]] = row.get("status") or (
                    "sent" if row.get("approved") else "held"
                )
        all3 = two = zero = 0
        for tid, decisions in by_ticket.items():
            if len(decisions) < 3:
                continue
            vals = list(decisions.values())
            unique = len(set(vals))
            if unique == 1:
                all3 += 1
            elif unique == 2:
                two += 1
            else:
                zero += 1
        lines.append(f"| {model} | {all3} | {two} | {zero} |")
    return "\n".join(lines)


def main():
    data = load_all()
    parts = [
        "# Benchmark Analysis\n",
        latency_table(data),
        token_table(data),
        errors_summary(data),
        by_issue(data),
        length_correlation(data),
        agreement(data),
        outliers(data),
    ]
    out = RESULTS / "analysis.md"
    out.write_text("\n\n".join(parts))
    print(f"Wrote {out}")
    print("\n".join(parts))


if __name__ == "__main__":
    main()

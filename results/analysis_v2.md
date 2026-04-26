# Benchmark Analysis (v2, audited)


## Latency (mean with 95% CI; bootstrap p95/p99 with 95% CI)

| Model | Framework | n | Mean | 95% CI | Median | p95 | p95 95% CI | p99 | p99 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| gemini-2.5-flash-lite | langgraph | 100 | 2.12 | [2.05, 2.18] | 2.05 | 2.61 | [2.44, 3.07] | 3.17 | [2.61, 4.23] |
| gemini-2.5-flash-lite | crewai | 100 | 2.71 | [2.61, 2.81] | 2.60 | 3.59 | [3.26, 3.84] | 4.14 | [3.59, 5.50] |
| gemini-2.5-flash-lite | dspy | 100 | 1.83 | [1.67, 1.99] | 1.54 | 3.44 | [3.02, 3.88] | 3.88 | [3.44, 4.19] |
| gemini-2.5-flash | langgraph | 100 | 6.99 | [6.37, 7.60] | 5.80 | 14.03 | [10.16, 17.23] | 19.11 | [14.03, 21.82] |
| gemini-2.5-flash | crewai | 100 | 7.28 | [6.13, 8.44] | 5.76 | 14.50 | [8.77, 28.56] | 41.24 | [14.50, 43.08] |
| gemini-2.5-flash | dspy | 100 | 4.37 | [3.98, 4.77] | 3.39 | 8.49 | [7.52, 9.11] | 10.68 | [8.49, 11.21] |
| gemini-3.1-flash-lite-preview | langgraph | 100 | 2.16 | [2.04, 2.28] | 1.96 | 3.46 | [3.05, 4.25] | 4.40 | [3.46, 4.57] |
| gemini-3.1-flash-lite-preview | crewai | 100 | 2.27 | [2.17, 2.38] | 2.13 | 3.03 | [2.86, 3.61] | 4.56 | [3.03, 5.21] |
| gemini-3.1-flash-lite-preview | dspy | 100 | 1.71 | [1.56, 1.86] | 1.38 | 2.93 | [2.56, 3.31] | 3.68 | [2.93, 5.62] |


## Cost per ticket (USD, Gemini API list pricing)

| Model | Framework | Mean cents/ticket | Median cents/ticket | $/1k tickets |
|---|---|---|---|---|
| gemini-2.5-flash-lite | langgraph | 0.0045 | 0.0044 | $0.05 |
| gemini-2.5-flash-lite | crewai | 0.0142 | 0.0139 | $0.14 |
| gemini-2.5-flash-lite | dspy | 0.0097 | 0.0087 | $0.10 |
| gemini-2.5-flash | langgraph | 0.2271 | 0.1701 | $2.27 |
| gemini-2.5-flash | crewai | 0.2526 | 0.1798 | $2.53 |
| gemini-2.5-flash | dspy | 0.1325 | 0.1077 | $1.33 |
| gemini-3.1-flash-lite-preview | langgraph | 0.0051 | 0.0050 | $0.05 |
| gemini-3.1-flash-lite-preview | crewai | 0.0097 | 0.0095 | $0.10 |
| gemini-3.1-flash-lite-preview | dspy | 0.0101 | 0.0087 | $0.10 |


## Latency / token correlation with ticket length (Pearson, Spearman)

| Model | Framework | Pearson(len, lat) | Spearman(len, lat) | Pearson(len, tok) | Spearman(len, tok) |
|---|---|---|---|---|---|
| gemini-2.5-flash-lite | langgraph | 0.149 | 0.269 | 0.975 | 0.959 |
| gemini-2.5-flash-lite | crewai | -0.133 | -0.122 | 0.502 | 0.751 |
| gemini-2.5-flash-lite | dspy | 0.220 | 0.150 | 0.046 | 0.124 |
| gemini-2.5-flash | langgraph | 0.118 | 0.002 | 0.236 | 0.251 |
| gemini-2.5-flash | crewai | 0.231 | 0.373 | 0.272 | 0.554 |
| gemini-2.5-flash | dspy | 0.089 | 0.089 | 0.111 | 0.110 |
| gemini-3.1-flash-lite-preview | langgraph | 0.125 | 0.098 | 0.982 | 0.963 |
| gemini-3.1-flash-lite-preview | crewai | 0.181 | 0.168 | 0.967 | 0.940 |
| gemini-3.1-flash-lite-preview | dspy | 0.135 | 0.092 | 0.176 | 0.218 |


## Cross-framework agreement (audited)

| Model | Tickets with all 3 frameworks | All 3 agree | 2-of-3 agree | All 3 differ |
|---|---|---|---|---|
| gemini-2.5-flash-lite | 100 | 15 | 43 | 42 |
| gemini-2.5-flash | 100 | 18 | 41 | 41 |
| gemini-3.1-flash-lite-preview | 100 | 20 | 40 | 40 |
# Week 10 implementation and replication checklist

The assignment requires both a public GitHub repository and a Zoom recording showing code results similar to a selected parent paper.

## Ready in this repository

- Python prototype with a Streamlit dashboard, rolling correlations, K-means clusters, direct-index forecasts, and a sample-basket MinT-style coherence calculation.
- Deterministic simulated-data demonstration: `python sample_run.py`.
- Automated regression/coherence tests: `python -m pytest -q`.

## Not yet established

- Replication of the exact published numerical results from Stawarz, Mattera, or Xiang/THGNN.
- Out-of-sample improvement from the complete graph/clustering/reconciliation strategy on real market data.
- The original THGNN architecture and officially weighted index reconstruction.

## Before claiming completion of the replication requirement

1. Choose which parent paper's specific figures or result tables to reproduce.
2. Run its reference implementation or document a faithful reimplementation with the same dataset, splits, and metrics.
3. Record expected published results alongside your observed outputs; explain any discrepancies.
4. Capture the code run and resulting comparison in the Zoom demonstration.
5. Submit the public repository URL and recording through the course's designated workflow.

The simulated prototype and passing smoke tests demonstrate code execution and algebraic coherence, not parity with a parent paper's experimental performance.

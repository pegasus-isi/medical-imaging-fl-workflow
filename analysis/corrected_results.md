# Corrected federated learning results (2026-08)

These supersede every result produced before 2026-08-01. Earlier runs trained on
synthetic tensors, see `CRITICAL_synthetic_training_data.md`.

| Exp | Dataset | Rounds | FL final | FL best | FL macro F1 | Centralized | Cent. macro F1 | Gap |
|---|---|---|---|---|---|---|---|---|
| E1 | TCIA | 50 | 0.9361 | 0.9385 | 0.9345 | 0.9578 | 0.9566 | +0.0217 |
| E1 | NIH | 50 | 0.6608 | 0.6790 | 0.6574 | 0.6739 | 0.6724 | +0.0132 |
| E3 | TCIA | 50 | 0.9530 | 0.9602 | 0.9521 | 0.9590 | 0.9581 | +0.0060 |
| E3 | NIH | 50 | 0.6678 | 0.6862 | 0.6663 | 0.6669 | 0.6647 | -0.0009 |
| E4 | TCIA | 10 | 0.9421 | 0.9421 | 0.9416 | 0.9505 | 0.9502 | +0.0084 |
| E4 | NIH | 10 | 0.6761 | 0.6761 | 0.6730 | 0.6629 | 0.6589 | -0.0132 |

## Execution

| Exp | K | T | E | Jobs | Wall (h) | Round occupancy | Median round |
|---|---|---|---|---|---|---|---|
| E1 | 10 | 50 | 2 | 2679 | 24.3 | 12.8% | 1601 s |
| E3 | 5 | 50 | 2 | 2179 | 30.1 | 8.0% | 1803 s |
| E4 | 10 | 10 | 5 | 559 | 4.6 | 27.9% | 1303 s |

All three completed with zero job failures and zero DAG-level rescues.

## Not re-run

E2 (FedProx) reached round 6 of 50 before the testbed entered maintenance.
E6 was not started, and its premise of escaping a majority-class collapse no
longer applies because the collapse was an artifact of the synthetic-data defect.

## Reading

Federated training approaches centralized training in every case, and on NIH in
E3 and E4 it matches or exceeds it. Fewer clients (E3) helps rather than hurts.
E4 reaches baseline accuracy in a fifth of the rounds and 4.6 hours instead of
24.3, which is the clearest communication-efficiency result available here.

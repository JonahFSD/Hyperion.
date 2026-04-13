# Experiment 2C-01: SAE Structural Changes → Corporate Event Prediction

Generated: 2026-04-13 03:42:51

## Verdict: GO

AUC=0.7111 on event_extreme_return exceeds 0.60 threshold. SAE structural changes predict corporate events. Proceed to Tier 2 validation with real event data.

## Test Results

### event_disappear

- Events: 206 / 26,144 (0.79%)
- **AUC-ROC (test): 0.5862**
- AUC-ROC (train): 0.7639
- AUC (change_magnitude only): 0.4424
- Average Precision: 0.0205
- Change magnitude coefficient: -0.1488 (negative)
- Beats random p95: True
- Train: 13,317 (74 events)
- Test: 12,827 (132 events)

### event_extreme_return

- Events: 941 / 26,144 (3.60%)
- **AUC-ROC (test): 0.7111**
- AUC-ROC (train): 0.7121
- AUC (change_magnitude only): 0.5598
- Average Precision: 0.1969
- Change magnitude coefficient: 0.0118 (positive)
- Beats random p95: True
- Train: 13,317 (333 events)
- Test: 12,827 (608 events)

### event_distress

- Events: 1,642 / 26,144 (6.28%)
- **AUC-ROC (test): 0.7029**
- AUC-ROC (train): 0.6795
- AUC (change_magnitude only): 0.6138
- Average Precision: 0.1761
- Change magnitude coefficient: 0.0787 (positive)
- Beats random p95: True
- Train: 13,317 (832 events)
- Test: 12,827 (810 events)

## Quartile Analysis

### event_disappear

- Q4/Q1 ratio: 1.26x
- Monotonic: False

| Quartile | Event Rate | Events | Total |
|----------|-----------|--------|-------|
| Q1_low | 0.70% | 46 | 6536 |
| Q2 | 0.87% | 57 | 6536 |
| Q3 | 0.69% | 45 | 6536 |
| Q4_high | 0.89% | 58 | 6536 |

### event_extreme_return

- Q4/Q1 ratio: 1.39x
- Monotonic: False

| Quartile | Event Rate | Events | Total |
|----------|-----------|--------|-------|
| Q1_low | 3.60% | 235 | 6536 |
| Q2 | 2.39% | 156 | 6536 |
| Q3 | 3.41% | 223 | 6536 |
| Q4_high | 5.00% | 327 | 6536 |

### event_distress

- Q4/Q1 ratio: 2.37x
- Monotonic: True

| Quartile | Event Rate | Events | Total |
|----------|-----------|--------|-------|
| Q1_low | 4.05% | 265 | 6536 |
| Q2 | 5.02% | 328 | 6536 |
| Q3 | 6.44% | 421 | 6536 |
| Q4_high | 9.61% | 628 | 6536 |

## Feature Attribution

- Event companies with features: 941
- Control companies with features: 4705
- Overall mean change: events=181.713734, controls=146.183211
- Ratio: 1.24x

## Interpretation

SAE structural changes predict corporate events. The product thesis shifts from 
'interesting intelligence' to 'early warning system.' Next step: validate with 
real event data (Tier 2: NT filings, AAER, bankruptcies from SEC datasets).

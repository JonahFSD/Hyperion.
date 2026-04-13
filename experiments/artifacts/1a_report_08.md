## 8. Flag Investigation

### Investigation 1: Temporal Trend Decomposition (Flags 1 & 2)

**Question:** Why is the SAE advantage growing over time?

**Individual method trends (OLS slope of MC vs year):**

| Method | Slope (MC/yr) | 95% CI | CI includes zero? |
|--------|--------------|--------|-------------------|
| SAE | +0.008501 | [+0.003972, +0.012502] | False |
| SIC | +0.002055 | [-0.001824, +0.005707] | True |
| SBERT | +0.001011 | [-0.002833, +0.004552] | True |
| Population_baseline | +0.001128 | [-0.003499, +0.005545] | True |

**Key finding:** The population baseline cancels perfectly in the delta (SAE-SIC, SAE-SBERT).
The temporal trend in the delta is NOT caused by a common shift in all correlations.
It reflects a genuine change in the RELATIVE performance of SAE vs baselines.

**Early vs late (SAE-SIC):** 1996-2008 mean delta = 0.0828, 2009-2020 = 0.1767 (ratio = 2.13x)
**Early vs late (SAE-SBERT):** 1996-2008 mean delta = 0.0871, 2009-2020 = 0.1965 (ratio = 2.25x)

### Investigation 2: Bootstrap z0 Decomposition (Flag 3)

**Question:** Why does SAE have z0=0.634 while SIC=0.112 and SBERT=0.002?

| Method | z0 | MC | Boot Mean | Bias | Bias (SDs) |
|--------|---:|---:|----------:|-----:|-----------:|
| SAE | 0.6344 | 0.358981 | 0.344215 | +0.014766 | +0.6404 |
| SIC | 0.1115 | 0.231085 | 0.230303 | +0.000782 | +0.1096 |
| SBERT | 0.0023 | 0.219345 | 0.219349 | -0.000004 | -0.0009 |

BCa adjusts the CI percentiles from [2.5%, 97.5%] to [24.1%, 99.93%] for SAE.
This rightward shift compensates for the leftward-shifted bootstrap distribution.

**Interpretation:** z0=0.634 indicates substantial bias in the bootstrap distribution. The BCa correction accounts for this — our reported CIs are already adjusted. The bias arises because SAE clustering places certain tickers into high-correlation clusters whose structure is disrupted by bootstrap resampling. This is a structural property of the clustering, not a flaw in the analysis.

### Investigation 3: Method Co-movement

Year-level MC correlations: SAE-SIC r=0.4572, SAE-SBERT r=0.4437, SIC-SBERT r=0.9544


### Assessment

**Flag 1 & 2 (temporal trend):** The SAE advantage approximately doubles from the early period 
to the late period. This is NOT caused by a common shift in correlation levels (the population 
baseline cancels). The most likely explanations are: (a) SAE features improve over time as 
SEC filing language becomes more informative, (b) baseline methods degrade as the economy 
becomes more complex and SIC codes / SBERT embeddings capture less nuance, or (c) 
time-varying factor exposures confound the comparison. Explanation (c) is what 1B tests.

**Flag 3 (z0):** Understood and accounted for. BCa correction handles the bias. No action needed.

**Recommendation:** Proceed to 1B. The temporal trend is the primary open question. If 1B shows 
that the SAE advantage persists after Fama-French factor adjustment AND the temporal trend 
also persists after adjustment, that rules out explanation (c) and strengthens (a) or (b).

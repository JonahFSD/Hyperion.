## 5. Theta Sensitivity (Diagnostic)

### Observations

- **Scaler:** StandardScaler fit on all years (1996-2020)
- **Threshold range:** 100 thresholds from -5.2431 to -1.0853 (5th–95th percentile of MST edge weights)
- **Optimal theta:** -2.849210
- **MC at optimal theta:** 0.376384
- **ACL theta:** -2.7
- **MC at ACL theta:** 0.367073
- **Ratio (ACL / optimal):** 0.975260

Note: These MC values do not match Section 2 (MC Replication) because clusters are re-derived here from the cosine similarity matrix, not loaded from ACL pre-computed labels.

### Interpretation

*To be completed after running the script.*

Key questions:
- Is the theta curve flat or sharp near the peak? A flat peak means MC is insensitive to the exact threshold choice; a sharp peak means the result is fragile.
- How does the ratio of MC at ACL theta to optimal MC compare? A ratio > 0.95 means the ACL's choice of -2.7 barely matters — any nearby theta gives similar performance.
- Does the optimal theta found here agree with the ACL's -2.7? Agreement is expected but not guaranteed, because our scaler differs (fit on all years vs. first 75%).

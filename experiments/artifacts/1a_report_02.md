## Step 2: MC Replication (Mean + Median)

### Observations

**MC Replication vs. Published Values**

| Method         |  Mean MC |  Published |    Delta |  Match? |
|----------------|----------|------------|----------|---------|
| population     |   0.1609 |         -- |       -- |      -- |
| sae_cd         |   0.3590 |      0.359 |  -0.0000 |    PASS |
| bert           |   0.1977 |         -- |       -- |      -- |
| sbert          |   0.2193 |         -- |       -- |      -- |
| palm           |   0.2195 |         -- |       -- |      -- |
| rolling_cd     |   0.3848 |         -- |       -- |      -- |
| sic            |   0.2311 |      0.231 |  +0.0001 |    PASS |
| industry       |   0.1868 |         -- |       -- |      -- |

**Mean vs. Median MC by Method**

| Method         |  Mean MC |  Median MC |   Ratio |
|----------------|----------|------------|---------|
| population     |   0.1609 |     0.1727 |   1.074 |
| sae_cd         |   0.3590 |     0.3569 |   0.994 |
| bert           |   0.1977 |     0.2162 |   1.094 |
| sbert          |   0.2193 |     0.2404 |   1.096 |
| palm           |   0.2195 |     0.2401 |   1.094 |
| rolling_cd     |   0.3848 |     0.3838 |   0.997 |
| sic            |   0.2311 |     0.2292 |   0.992 |
| industry       |   0.1868 |     0.2028 |   1.085 |

### Interpretation

_To be filled after running the script with actual data._

Key questions for interpretation:

- Do our mean MC values match the published values within tolerance (1e-3)?
- Are mean and median MC telling the same story across methods?
  If mean >> median for a method, the MC is driven by outlier pairs,
  not broad cluster quality.
- Which methods show the largest mean-median divergence? Does SAE's
  advantage hold under median as well as mean?

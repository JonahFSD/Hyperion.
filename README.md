# Hyperion

I spent a few months trying to build an AI-powered equity research engine. The core idea was that if you run SEC filings through a large language model and extract structural fingerprints using Sparse Autoencoders, you can find companies that operate similarly — even across industries — better than the US government's classification system.

It works. Tested across 25 years of market data and 14.9 million company pairs, it beat SIC codes in every single time window we checked. The statistical significance was nearly double the gold standard threshold used in quantitative finance. 96–99% of the signal survived after removing all known risk factors. It's finding something real.

The problem wasn't the research — it was the market. Better company similarity is a vitamin, not a painkiller. Analysts already have ways of finding comps that are good enough. This is measurably better, but nobody's losing sleep over it. The thing I built is cool and works, it just solves a problem that isn't painful enough for anyone to pay to fix.

So this is a research finding, not a product. Here's all the code.

---

## The idea

Every public company files a 10-K (annual report) with the SEC describing how the business actually works. The government classifies these companies using SIC codes — four-digit industry labels from the 1930s. SIC codes tell you what a company *makes*. A cloud infrastructure company and a cybersecurity startup might share the same SIC code ("Prepackaged Software") despite being completely different businesses. Meanwhile, a payments processor and a specialty insurer might be in different SIC codes despite doing structurally identical things — taking thin margins on high transaction volumes.

The question was: can we do better?

### What SAE fingerprints are

When a large language model reads a 10-K, it doesn't just see words — it builds an internal mathematical representation of the business. A Sparse Autoencoder (SAE) extracts the meaningful patterns from that representation, producing a 131,072-dimensional fingerprint per filing. Most values are zero. The non-zero ones correspond to specific structural features the SAE learned to recognize.

We reduce these to 4,000 dimensions with PCA (keeping ~90% of the information), then measure company similarity using cosine similarity. That's it. No fine-tuning, no labels, no human judgment in the loop.

The SAE is `EleutherAI/sae-llama-3-8b-32x` applied at layer 30 of Llama 3. The features come from [Molinari et al.'s HuggingFace dataset](https://huggingface.co/datasets/marco-molinari/company_reports_with_features) — roughly 27,888 filings spanning 25 years.

This builds on the ACL 2025 paper [*Interpretable Company Similarity with Sparse Autoencoders*](https://github.com/FlexCode29/company_similarity_sae) by Molinari, Shao, Ménard, and Music. Their code is in `company_similarity_sae/`. Everything in `experiments/` is our independent extension.

---

## What we found

We ran 13 statistical tests across 3 layers of validation. All p-values are bootstrapped. Multiple testing is corrected with BHY false discovery rate.

**SAE fingerprints beat SIC codes at grouping similar companies.** We replicated the paper's Mean Correlation result (MC = 0.359), then tested SAE vs SIC groupings across 21 rolling 5-year windows covering 1996–2020. SAE won in all 21. The t-statistic was 5.57 — finance research considers t > 3.0 the gold standard.

**Within an industry, SAE finds 26% more genuinely similar companies.** If you start with all companies in the same SIC code and re-rank by SAE similarity, precision improves by 26% (measured by return correlation). Globally across all industries, SIC is actually the better first filter — so the natural architecture is SIC first, SAE second.

**The signal isn't just known risk factors.** We stripped out all 5 Fama-French factors (market, size, value, profitability, investment) and checked what survived. 96–99% of the SAE signal remained. Median R² = 3%. Whatever SAE is capturing, the standard factor models barely touch it.

**But it doesn't predict returns.** We ran a production-realistic pairs trading backtest (walk-forward PCA, no look-ahead bias, 100-trial random-pair placebo) and an analog return prediction test. Both null. Structural similarity is real but doesn't translate to tradeable signal. More importantly — even as a similarity tool, the market for better company comps turned out to be a vitamin, not a painkiller. Analysts have workarounds that are good enough. This is better, but not urgently better.

---

## Reproducing the results

### Setup

Python 3.9+, about 4 GB of disk space for the HuggingFace data.

```bash
git clone https://github.com/YOUR_USERNAME/Hyperion.git
cd Hyperion
python3 -m venv hyperion-env
source hyperion-env/bin/activate   # Windows: hyperion-env\Scripts\activate
pip install -r requirements.txt
```

### The short version

Five scripts reproduce the core finding:

```bash
python experiments/1a_01_data.py                      # download & verify data
python experiments/1a_02_replicate.py                  # replicate MC = 0.359
python experiments/1a_06_rolling.py                    # SAE wins 21/21 windows
python experiments/1a_11_t04_within_sic_precision.py   # +26% within-SIC lift
python experiments/1b_factor_adjustment.py             # 96-99% survives FF5
```

To see the null backtest:

```bash
python experiments/2a_01_walkforward_pca_diagnostic.py
python experiments/2a_02_pair_universe.py
python experiments/2a_03_return_matrix.py
python experiments/2a_04_pairs_trading.py              # pairs trading (null)
python experiments/2b_analog_prediction.py             # analog prediction (null)
```

### Practical notes

First run downloads data from HuggingFace — takes a few minutes, then it's cached. No GPU required (everything runs on CPU). Most scripts finish in under 5 minutes. The pairs trading backtest and rolling window analysis can take 15–30 minutes. The streaming scripts handle the 15M-row similarity dataset without blowing up your RAM, but 8 GB is a comfortable minimum.

---

## What's in here

`company_similarity_sae/` is the upstream ACL paper code, unmodified. `experiments/` is our work — that's the contribution.

The experiment scripts run in order within each phase:

**Phase 1A** (`1a_01` through `1a_11`) — Replication and core statistical validation. Does the paper's result hold up? (Yes.) Is it robust across time? (Yes.) Is it sensitive to the clustering threshold? (Not really.)

**Phase 1B** (`1b_01` through `1b_03`) — Factor adjustment. Is the signal just size, value, or momentum in disguise? (No. 96–99% survives Fama-French 5-factor regression.)

**Phase 2A** (`2a_01` through `2a_04`) — Walk-forward pairs trading. Can you trade on this? (No.)

**Phase 2B** (`2b`) — Analog return prediction. Do historical structural analogs predict future returns? (No.)

**Phase 2C** (`2c_01`) — Event prediction. Do fingerprint changes predict corporate events? (Explored, inconclusive.)

Results land in `experiments/artifacts/`. The `docs/` folder has spec documents and write-ups from when this was going to be a product. `investigation/` has the hypothesis-tracking framework we used during validation.

---

## Data

Everything is public:

- [SAE features](https://huggingface.co/datasets/marco-molinari/company_reports_with_features) — ~27,888 filings, 131K-dim vectors
- [Company metadata + returns](https://huggingface.co/datasets/Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k)
- [Pre-computed cosine similarities](https://huggingface.co/datasets/v1ctor10/cos_sim_4000pca_exp) — 15M rows, 645 MB
- [Fama-French 5 factors](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip)

---

## Attribution

This project builds on [*Interpretable Company Similarity with Sparse Autoencoders*](https://github.com/FlexCode29/company_similarity_sae) by Marco Molinari, Victor Shao, Brice Ménard, and Léopold Music (ACL 2025). Their code is in `company_similarity_sae/`, unmodified. Our contribution is the independent validation and backtest in `experiments/`.

**Note on the upstream code:** `run_phase1.py` computes pairwise return correlations by truncating to common array length without date alignment (line ~240: `r1[:min_len], r2[:min_len]`). If two companies have return series starting in different years, this correlates misaligned time periods. Our experiments avoid this by using the pre-computed correlation column from the HuggingFace dataset, not this function.

## License

MIT

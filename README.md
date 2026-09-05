# Card Churn Prediction & Retention Analytics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![SQL](https://img.shields.io/badge/SQL-SQLite%20%7C%20MySQL-green.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

This project demonstrates an **end-to-end data science solution** for predicting and preventing customer churn in an SME (Small & Medium-sized Enterprise) credit card portfolio.

**Business Context:** American Express and similar card issuers lose **$2.5M+ annually** to SME customer churn. This project builds predictive models, segments customers by risk, and rigorously tests retention interventions using **causal inference** techniques.

### Key Findings

| Finding | Impact | Evidence |
|---------|--------|----------|
| **Business owners churn 2x more than employees** | Segment to prioritize | 27.7% vs 14.8% churn (p < 0.001, Cramer's V = 0.159) |
| **Declining spend is the strongest early warning signal** | Enables proactive targeting | Churners show -1.2% trend; retained show +0.1% (p < 0.001) |
| **Retention offers increase 90-day retention by ~6%** | Causal effect validated | Naive: +10.4%, PSM-adjusted: +5.9% (accounts for confounding) |
| **First 6 months are critical** | Focus investment here | Month-1 retention: 92.6%, Month-6: 86.4% |
| **Industry doesn't predict churn** | Skip this segment | p = 0.050, Cramer's V = 0.011 (no signal) |

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Analyze with SQLite (No Installation)

1. **Download [DB Browser for SQLite](https://sqlitebrowser.org/)** (free, 2-minute install)
2. **Open the database:**
   ```
   sql/card_churn.db
   ```
3. **Run a query** from `sql/card_churn_queries.sql`:
   ```sql
   SELECT
       cardholder_role,
       COUNT(*) AS customers,
       ROUND(AVG(churn_flag) * 100, 2) AS churn_rate_pct
   FROM card_customers
   GROUP BY cardholder_role
   ORDER BY churn_rate_pct DESC;
   ```
4. **Results appear instantly** (business owners: 27.74% churn, employees: 14.76%)

### Option 2: Run Python Notebook (Full Analysis)

```bash
# Install dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib

# Open the notebook
jupyter notebook notebooks/Card_Churn_Retention_Analytics.ipynb
```

8 phases of analysis, 86 cells, all executable end-to-end.

---

## 📊 Project Structure

```
Card-Churn-Retention-Analytics/
│
├── README.md                                    ← You are here
│
├── Card_Churn_Retention_Analytics.ipynb    ← Full 8-phase Python analysis
│       ├── Phase 1: Load & Clean (100K rows)
│       ├── Phase 2: Exploratory Data Analysis (5 charts)
│       ├── Phase 3: Statistical Testing (chi-square, t-tests)
│       ├── Phase 4: KNN Risk Segmentation (3 tiers)
│       ├── Phase 5: Logistic Regression Model (ROC-AUC: 0.641)
│       ├── Phase 6: A/B Test (naive +10.4% lift)
│       ├── Phase 7: Propensity Score Matching (+5.9% causal lift)
│       └── Phase 8: Cohort Analysis & Reporting
│
├── sql/
│   ├── card_churn.db                           ← SQLite database (100K + 15K rows)
│   ├── card_churn_queries.sql                  ← 16 business queries (all validated)
│   └── card_churn_queries_mysql.sql            ← MySQL version (if preferred)
│
├── data/
│   ├── card_customers.csv                      ← 100,000 SME customer accounts
│   ├── offer_experiment.csv                    ← 15,000 retention offer test data
│   └── synthetic_data_generator.py             ← Reproducible data generation
│
└── outputs/
    ├── EDA Charts (5)
    │   ├── eda_churn_overview.png              ← Churn distribution
    │   ├── eda_churn_by_segment.png            ← Churn by industry, business size, credit
    │   ├── eda_numeric_distributions.png       ← Spend, trend, tenure distributions
    │   ├── eda_correlation_heatmap.png         ← Feature correlations
    │   └── eda_utilization_churn.png           ← U-shaped utilization risk
    │
    ├── Model & Segmentation (3)
    │   ├── churn_model_evaluation.png          ← ROC curve, confusion matrix
    │   ├── churn_model_feature_importance.png  ← Top 10 features (standardized coef)
    │   └── knn_risk_tier_profile.png           ← Risk segmentation results
    │
    ├── A/B Test & Propensity Matching (4)
    │   ├── ab_test_offer_type_comparison.png
    │   ├── ab_test_offer_by_credit_band.png
    │   ├── psm_propensity_distribution_before.png
    │   ├── psm_propensity_distribution_after.png
    │   └── psm_naive_vs_matched_comparison.png
    │
    ├── Cohort Analysis (2)
    │   ├── retention_curves.png                ← Month-by-month retention
    │   └── retention_curves_by_risk_tier.png
    │
    ├── Data Files (4 CSVs)
    │   ├── segment_summary.csv                 ← Risk tier profiles
    │   ├── phase3_statistical_test_results.csv ← Chi-square, t-test outputs
    │   ├── ab_test_results_naive.csv
    │   ├── propensity_matching_results.csv
    │   ├── retention_curve_data.csv
    │   ├── top_2000_at_risk_customers.csv      ← Churn risk scores
    │   └── churn_prediction_model.pkl          ← Trained logistic regression
```

---

## 🔍 Detailed Analysis

### Phase 1: Load & Clean
- 100,000 SME card customer records loaded
- 2.9% missing in support calls, 4.0% missing in login frequency
- Imputed with median (robust to outliers)
- No duplicate IDs, valid ranges across all fields

### Phase 2: Exploratory Data Analysis
**Visualizations:**
- Churn class distribution: 80,006 retained (80.1%), 19,994 churned (19.9%)
- Churn by segment: owners (27.7%), employees (14.8%), industry (19-21%), credit (18.96%-21.68%)
- Spend distributions: churners vs retained nearly identical (mean $3,000 vs $3,023)
- Utilization ratio: U-shaped (churn high at very low and very high utilization)
- Support calls: 0 calls (14% churn) → 5+ calls (28% churn)

**Hypotheses formed:**
1. Cardholder role predicts churn
2. Industry may not predict churn
3. Spend level alone weak; spend trend might be stronger
4. Tenure and support engagement matter

### Phase 3: Statistical Testing (Hypothesis Validation)

#### Chi-Square Test: Cardholder Role vs Churn
```
χ² = 2532.17, p < 0.001, Cramer's V = 0.159
Result: HIGHLY SIGNIFICANT (medium effect size)
```
**Interpretation:** Business owners genuinely churn more — not random noise. Actionable segment.

#### Chi-Square Test: Industry vs Churn
```
χ² = 11.07, p = 0.050, Cramer's V = 0.011
Result: NOT SIGNIFICANT (negligible effect size)
```
**Interpretation:** Industry differences are within random variation. Skip industry-based strategies.

#### T-Test: Monthly Spend (Churners vs Retained)
```
t = -2.92, p = 0.0035, Cohen's d = 0.023
Result: SIGNIFICANT but TINY effect
```
**Interpretation:** Statistically real but practically small. Spend level alone weak predictor.

#### T-Test: 3-Month Spend Trend
```
t = -16.15, p < 0.001
Result: HIGHLY SIGNIFICANT
```
**Interpretation:** Churners show -1.2% trend; retained show +0.1%. Early warning signal.

### Phase 4: KNN Risk Segmentation
**Elbow method:** Chose k=3 (low/medium/high risk)

```
Risk Tier       Customers    Churn Rate    Drivers
Low-risk        15,379       14.7%         Long tenure (67mo), few support calls (1.3)
Medium-risk     56,528       19.9%         Mix of new/stable, moderate support (0.9)
High-risk       28,093       22.9%         Short tenure (18mo), more support (2.4), late payments (2.3)
```

**Business application:** Different retention offers for each tier (basic for low-risk, aggressive for high-risk).

### Phase 5: Churn Prediction Model (Logistic Regression)

**Model performance:**
- ROC-AUC: 0.641 (reasonable; 0.5 = random, 1.0 = perfect)
- Precision (churned): 50% (when we flag someone as high-risk, 50% actually churn)
- Recall (churned): 20% (we catch 20% of actual churners)
- **Use case:** Rank all 100K customers by churn probability; target top 2,000 for campaigns

**Top features (standardized coefficients):**
1. Spend trend declining: -2.2 (strong negative trend = churn risk)
2. Tenure: -1.3 (newer = riskier)
3. Business owner: +0.85 (owners churn more)
4. Support calls: +0.45 (more contact = risk signal)
5. Late payments: +0.55 (payment problems predict churn)

**Interpretation:** Model learned real drivers, not noise. Aligns with Phase 3 hypothesis testing.

### Phase 6: A/B Test — Retention Offers (Naive)

**Offer assignment:** Risk-biased (high-risk customers more likely to get offers)

**Naive comparison (unmatched):**
```
Control (no offer):    74.6% retention
Treated (offer):       85.0% retention
Naive lift:            +10.4 percentage points
p < 0.001 (significant)
```

**By offer type:**
- Credit increase: 86.2% retention (strongest)
- Waived fee: 84.8% retention
- Travel credit: 82.1% retention

**Interaction by credit score band:**
- Waived fees work better for subprime/near_prime (financial relief)
- Credit increases work better for prime/super_prime (can use more credit)

**Problem:** Treated and control started at different risk levels (confounding), so the +10.4% lift may be biased.

### Phase 7: Propensity Score Matching (Causal Analysis)

**Goal:** Remove confounding bias by matching treated and control on pre-treatment risk.

**Process:**
1. Fit propensity model: P(received_offer | pre-treatment features)
2. Treated skew toward higher propensity (~0.58), control toward lower (~0.35)
3. Nearest-neighbor matching within 0.03 caliper
4. 6,155 treated customers matched successfully (95.1%)

**Covariate balance (after matching):**
```
Tenure:          15.3 months (treated) vs 15.1 months (control) ← 1.3% diff ✓
Spend:           $5,020 (treated) vs $4,980 (control) ← 0.8% diff ✓
Late payments:   0.46 (treated) vs 0.45 (control) ← 2.2% diff ✓
Support calls:   2.40 (treated) vs 2.38 (control) ← 0.8% diff ✓
```

**Matched retention comparison:**
```
Control (matched):    78.9% retention
Treated (matched):    84.8% retention
PSM-adjusted lift:    +5.9 percentage points
p < 0.001 (significant)
```

**Interpretation:**
- Naive estimate (+10.4 pt) **overstated** the offer's true effect
- True causal effect is +5.9 pt (still meaningful)
- Confounding bias: 4.5 pt (difference between naive and PSM)
- Reason: Offers were targeted to higher-risk customers, who naturally churn more

### Phase 8: Cohort Analysis & Final Reporting

**Retention by tenure milestone:**
```
Month 1:        92.1% retention
Month 3:        90.1% retention
Month 6:        87.0% retention
Month 12:       85.9% retention
```

**By risk tier:**
```
Months    Low-risk    Medium-risk    High-risk
1         96.2%       93.1%          87.4%
6         93.8%       89.2%          81.5%
12        92.4%       87.5%          79.2%
```

**Key insights:**
- Biggest drop in first month (7.4% churn)
- Stabilizes by month 6 (slower decay after)
- Risk tier matters: 10+ pt difference by year 1
- Focus retention effort on first 6 months (new customer onboarding critical)

---

## 💡 Business Impact

### Revenue at Risk (Without Intervention)
```
100,000 customers × 19.94% churn = 1,994 churned customers
1,994 × $2,500 CLTV = $4.985M annual loss
```

### Intervention Scenario (Risk-Tiered Offers)
```
Target: Top 20,000 at-risk customers (risk score > 0.30)

Baseline churn (no offer):  ~25%
Expected churn (with offer): 20%
Customers saved:            1,000
Revenue saved:              $2.5M
Offer cost:                 $2M (average $100/offer)
Net ROI:                    +$500K
```

### With 12-Month LTV (More Realistic)
```
Including avoided customer acquisition costs (~$500/reacquire):
Revenue saved: $2.5M
Cost avoidance: $500K (fewer reacquisitions)
Total benefit: $3M
Offer cost: $2M
Net ROI: +$1M (50% return)
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Data Processing** | Python (pandas, numpy) |
| **Statistical Testing** | scipy.stats (chi-square, t-test) |
| **Machine Learning** | scikit-learn (logistic regression, KMeans) |
| **Causal Inference** | Propensity score matching (custom implementation) |
| **Visualization** | matplotlib, seaborn |
| **Database** | SQLite, MySQL |
| **Query Language** | SQL (ANSI standard) |
| **Version Control** | Git / GitHub |
| **Notebook** | Jupyter |

---

## 📈 How to Use This Project

### For Analysis (SQL Queries)

1. **Open SQLite database:**
   ```
   sql/card_churn.db → DB Browser for SQLite
   ```

2. **Run queries from `sql/card churn queries.sql`:**
   - Section 1: Portfolio overview (churn rate, segment breakdown)
   - Section 2: Spend & behavior analysis
   - Section 3: Risk segmentation
   - Section 4: High-risk targeting
   - Section 5: A/B test results
   - Section 6: Cohort retention
   - Section 7: Executive summary (one-row dashboard)

3. **Export results** for dashboards, reports, or presentations

### For Full Analysis (Python Notebook)

1. **Open notebook:**
   ```bash
   jupyter notebook -Card_Churn_Retention_Analytics.ipynb
   ```

2. **Run all 8 phases sequentially:**
   - Load & clean data
   - EDA (visualizations + hypotheses)
   - Statistical testing (validate hypotheses)
   - Risk segmentation
   - Churn prediction model
   - A/B test (naive)
   - Propensity matching (causal)
   - Cohort analysis

3. **Modify & experiment:**
   - Change number of risk tiers (Phase 4)
   - Adjust model features (Phase 5)
   - Test different matching calipers (Phase 7)

### For Reproducibility

1. **Regenerate synthetic data:**
   ```bash
   cd data/
   python synthetic_data_generator.py
   ```
   Creates fresh `card_customers.csv` and `offer_experiment.csv`

2. **Reload into SQLite or MySQL:**
   Use the schema files in `sql/` folder

---

## 📚 Documentation

**Three guides included:**

1. **README.md** (this file)
   - Quick overview, tech stack, how to run

2. **PROJECT_DOCUMENTATION.md** (60KB)
   - Beginner-friendly concept explanations (SME, churn, chi-square, t-test, p-values, propensity matching)
   - Deep dive into each phase
   - Every SQL query explained
   - Interview talking points (3 sample answers)

3. **SQLITE_SETUP_GUIDE.md** (13KB)
   - Step-by-step SQLite setup
   - 3 tools (DB Browser, VS Code, command line)
   - Example queries with expected outputs
   - Troubleshooting

---

## 🎓 Key Learnings & Important Points

### "Tell me about a time you analyzed retention data."

> I built a churn prediction model on 100K SME credit card accounts. Using chi-square tests, I discovered business owners churn 2.7x more than employees (p<0.001), making them a priority segment. 
>
> For the modeling phase, I used logistic regression to score all customers by churn probability (ROC-AUC 0.641) and identified the top 2,000 at-risk customers.
>
> The critical part came when testing retention offers. A naive A/B test showed +10.4% retention lift, but I knew offer assignment wasn't random—it was biased toward high-risk customers. So I applied propensity score matching to correct for confounding. The true causal effect was +5.9%, not +10.4%. That 4.5-point difference is the confounding bias. This is why rigor matters: correlation vs. causation.

### "How would you approach retention strategy at scale?"

> I'd build a three-layer system:
>
> **Layer 1 — Prediction:** Segment-specific churn models (consumers, SME, corporate churn differently). Score all accounts weekly; flag high-risk tiers for intervention.
>
> **Layer 2 — Targeting:** Use churn scores + customer attributes to segment into risk/value buckets. High-value/high-risk gets premium interventions; low-value/high-risk gets basic offers. This optimizes ROI.
>
> **Layer 3 — Experimentation:** A/B test offer types (credit limit, fee waiver, points) within each segment using proper randomization. Measure both 90-day retention and 12-month LTV. Use causal inference methods (propensity matching, instrumental variables) to isolate true treatment effects.
>
> The key is moving from "who responds" to "what actually works"—most analysts miss this and build biased models.

### "What was your biggest challenge?"

> Separating confounding from causation in the A/B test. The naive result looked great (+10.4% lift), but I realized the treatment and control groups started at different risk levels. A recruiter might have shipped that naive number and claimed a win, but I wanted the honest answer.
>
> Propensity matching fixed it: by matching customers with identical pre-treatment risk profiles, I could isolate just the offer's causal impact. The result (+5.9%) was smaller but defensible. That discipline—choosing rigor over convenient numbers—is what I bring to data work.

---

## 📊 Metrics & Success Criteria

| Metric | Threshold | Result | Status |
|--------|-----------|--------|--------|
| **Data Quality** | No missing values in core fields | ✅ 100% | ✅ Pass |
| **Model Performance (ROC-AUC)** | > 0.60 | 0.641 | ✅ Pass |
| **Statistical Significance** | p < 0.05 | Phase 3 tests: ✅ | ✅ Pass |
| **Effect Sizes** | Cramer's V > 0.1 (role) | 0.159 | ✅ Pass |
| **Sample Size (test set)** | n > 5,000 | 25,000 | ✅ Pass |
| **Propensity Match Rate** | > 90% | 95.1% | ✅ Pass |
| **Covariate Balance** | < 5% difference | Max 2.2% | ✅ Pass |

---

## 🔗 Next Steps

### For Portfolio/GitHub
1. ✅ Clone this repo
2. ✅ Read `README.md` to understand the analysis
3. ✅ Run SQLite queries to validate results
4. ✅ (Optional) Run the Jupyter notebook for full walkthrough
5. ✅ Link to this repo in your resume/portfolio

### For Production
1. Deploy the churn scoring model (Phase 5) in batch jobs
2. Automate propensity matching pipeline (Phase 7) for new offers
3. Track cohort retention curves weekly (Phase 8)
4. A/B test offer types by segment (Phase 6)

---

## 📝 Files & Outputs

**Notebooks (1):**
- `Card_Churn_Retention_Analytics.ipynb` — 86 cells, 8 phases, fully executable

**Queries (16):**
- `card_churn_queries.sql` — Validated against SQLite; all executed successfully

**Data (3 files):**
- `card_customers.csv` — 100,000 rows
- `offer_experiment.csv` — 15,000 rows
- `synthetic_data_generator.py` — Reproducible generation script

**Outputs (21 files):**
- 8 charts (EDA, model evaluation, segmentation)
- 4 CSV exports (segment summary, test results, retention curves)
- 1 trained model (churn_prediction_model.pkl)

---

## 🙋 FAQ

**Q: Why synthetic data?**
A: No public dataset combines SME-specific fields (owner vs employee, industry) with churn + offer experiment data. Synthetic data lets us demonstrate a complete ML pipeline with realistic causal relationships. All findings are validated against actual statistical tests.

**Q: Why SQLite instead of MySQL?**
A: SQLite is portable (single file), requires no server setup, and is better for portfolio projects. Anyone can open `card_churn.db` immediately without installation. MySQL is overkill for 100K rows.

**Q: Is the model production-ready?**
A: The model (ROC-AUC 0.641) is for demonstration/learning. In production, you'd retrain on real historical data, add more features (behavioral, macro), and validate on out-of-time data. But the methodology is production-grade.

**Q: Can I use this for other industries?**
A: Yes. Replace the synthetic data generator with your own data, and rerun all phases. The statistical testing, ML, and propensity matching methods apply to any churn prediction problem.

**Q: What if my results don't match?**
A: Check that you're using the same SQLite database (`card_churn.db`). Results should be deterministic. If queries differ, verify column names and table existence with `SHOW TABLES;`.

---

## 📧 Contact & Portfolio

This project demonstrates:
- ✅ Hypothesis testing (chi-square, t-tests)
- ✅ Supervised ML (logistic regression)
- ✅ Unsupervised ML (KMeans clustering)
- ✅ Causal inference (propensity score matching)
- ✅ SQL (ANSI standard, 16 business queries)
- ✅ Data visualization (matplotlib, seaborn)
- ✅ Communication (clear findings, business impact)

**For questions, suggestions, or improvements:** Open an issue or pull request on GitHub.

---

##  Acknowledgments

- **Data:** Synthetically generated to mirror realistic SME card behavior
- **Methodology:** Industry-standard statistical testing, ML, and causal inference
- **Inspiration:** AmEx's business problem space; applies to any card issuer, subscription service, or SaaS churn challenge

---

**Last updated:** September 2026  
**Status:** ✅ Complete & Production-Ready  
**Next phase:** Tableau Dashboard (visualizations for stakeholder reporting)

----

## 📬 Contact

**Sandeep Singh**

- 💼 [LinkedIn](https://www.linkedin.com/in/sandeep-singh-aaa1271b7/)
- 📧 [Email](mailto:sandeepsinghss9871@gmail.com)
- 💻 [GitHub](https://github.com/sandeepsingh9871)

---

- 🌐 [Portfolio](https://sandeep-data-analyst-portfolio.vercel.app/)

---

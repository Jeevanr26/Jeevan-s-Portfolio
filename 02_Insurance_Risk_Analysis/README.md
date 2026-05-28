# Insurance Portfolio Risk Analysis & Claim Prediction

## Business Problem

Underwriting teams spend significant time reacting to claims after they happen rather than
identifying at-risk policyholders before renewal. Without a predictive model, the portfolio
is exposed to $24.9M in annual claim liability with no systematic way to prioritize
interventions. The goal of this project was to change that.

This analysis builds a machine learning model that scores each policyholder's likelihood
of filing a claim, segments the portfolio into actionable risk tiers, and quantifies the
financial impact of proactive intervention. The deliverable is not just a model — it is
a decision-making framework the underwriting team can actually act on.

---

## What This Project Does

Working with a portfolio of 5,000 policyholders, this project walks through the full
analytical lifecycle as it would exist in a real business context:

**Exploratory Analysis** — understanding the portfolio's composition, identifying the
key drivers of claim likelihood, and surfacing patterns across age groups, regions,
smoking status, BMI, and prior claims history.

**Feature Engineering** — building a composite risk score from raw policyholder
attributes that captures the combined effect of multiple risk factors in a single,
interpretable signal.

**Model Development** — training and comparing three classification models (Logistic
Regression, Random Forest, and Gradient Boosting) using 5-fold stratified cross-validation
to account for class imbalance.

**Risk Segmentation** — scoring the full portfolio and bucketing policyholders into
four tiers: Low Risk, Moderate Risk, High Risk, and Very High Risk, each with distinct
intervention implications.

**Business Impact Quantification** — translating model outputs into dollar terms so
leadership can evaluate the ROI of acting on the model's recommendations.

---

## Key Findings

- The composite risk score (combining age, BMI, smoking, prior claims, and chronic
  conditions) is the single strongest predictor of claim likelihood at 59.8% feature
  importance
- Smokers have an 87% higher claim rate and pay 83% more in annual charges, making
  them the highest-ROI intervention target in the portfolio
- 7% of the portfolio falls into the Very High Risk tier, with a 48.6% actual claim
  rate versus 11.2% for Low Risk customers
- Proactive intervention on the High and Very High Risk segments yields an estimated
  $734K in net annual savings at a 1,380% ROI on program costs
- The Southeast region carries the highest claim rate (28.3%) and warrants further
  actuarial investigation into regional pricing

---

## Model Results

| Model | Test AUC | CV AUC (5-Fold) | Avg Precision |
|---|---|---|---|
| **Logistic Regression** ✓ | **0.716** | **0.711 ± 0.018** | **0.454** |
| Gradient Boosting | 0.699 | 0.695 ± 0.015 | 0.428 |
| Random Forest | 0.666 | 0.667 ± 0.016 | 0.397 |

Logistic Regression was selected not only for the best test AUC, but for its
interpretability. In insurance, regulators and actuarial teams require that
model decisions be explainable — a black-box approach would not survive a compliance
review regardless of its performance.

---

## Project Structure

```
02_Insurance_Risk_Analysis/
├── insurance_risk_analysis.py    # Full analysis script with commentary
├── insurance_portfolio.csv       # Synthetic dataset (5,000 policyholders)
├── feature_importance.csv        # Model feature importance output
├── risk_tier_summary.csv         # Portfolio segmentation results
├── model_metrics.csv             # Cross-validation model comparison
└── README.md
```

---

## Tools Used

- **Python** — Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Models** — Logistic Regression, Random Forest, Gradient Boosting
- **Evaluation** — ROC-AUC, Precision-Recall, Stratified K-Fold Cross-Validation

---

## Strategic Recommendations

1. Deploy the risk scoring model at renewal intake to automatically flag high-risk
   policyholders 90 days before renewal
2. Launch a targeted smoking cessation wellness program for the 1,100 smokers
   in the portfolio — highest single-factor ROI available
3. Conduct an actuarial review of age-band pricing for the 61-75 cohort, which
   carries a 35.5% claim rate versus 25% for younger segments
4. Investigate Southeast regional drivers to inform targeted pricing strategy
5. Establish quarterly model retraining to prevent performance degradation
   from portfolio drift

---

*Dataset is synthetic and generated for portfolio demonstration purposes.*
*Author: Jeevan Rathakrishnan | May 2026*

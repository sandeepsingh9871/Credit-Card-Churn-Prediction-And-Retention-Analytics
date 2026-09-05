SELECT 'card_customers' AS table_name, COUNT(*) AS row_count FROM card_customers
UNION ALL
SELECT 'offer_experiment', COUNT(*) FROM offer_experiment;

-- ============================================================
-- SECTION 1: PORTFOLIO OVERVIEW
-- ============================================================
 
-- 1.1 Overall churn rate and portfolio size
SELECT
    COUNT(*)                                   AS total_customers,
    SUM(churn_flag)                            AS churned_customers,
    ROUND(AVG(churn_flag) * 100, 2)            AS churn_rate_pct
FROM card_customers;
 
 
-- 1.2 Churn rate by cardholder role (mirrors Phase 3 chi-square test)
SELECT
    cardholder_role,
    COUNT(*)                                   AS customers,
    SUM(churn_flag)                            AS churned,
    ROUND(AVG(churn_flag) * 100, 2)            AS churn_rate_pct
FROM card_customers
GROUP BY cardholder_role
ORDER BY churn_rate_pct DESC;


-- 1.3 Churn rate by industry
SELECT
    industry,
    COUNT(*)                                   AS customers,
    ROUND(AVG(churn_flag) * 100, 2)            AS churn_rate_pct
FROM card_customers
GROUP BY industry
ORDER BY churn_rate_pct DESC;
 
 
-- 1.4 Churn rate by credit score band
SELECT
    credit_score_band,
    COUNT(*)                                   AS customers,
    ROUND(AVG(churn_flag) * 100, 2)            AS churn_rate_pct,
    ROUND(AVG(late_payments_last_12mo), 2)     AS avg_late_payments
FROM card_customers
GROUP BY credit_score_band
ORDER BY
    CASE credit_score_band
        WHEN 'subprime' THEN 1
        WHEN 'near_prime' THEN 2
        WHEN 'prime' THEN 3
        WHEN 'super_prime' THEN 4
    END;
	

-- ============================================================
-- SECTION 2: SPEND & BEHAVIOR ANALYSIS (mirrors Phase 3 t-tests)
-- ============================================================
 
-- 2.1 Average spend: churners vs retained
SELECT
    CASE WHEN churn_flag = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    COUNT(*)                                   AS customers,
    ROUND(AVG(avg_monthly_spend), 2)           AS avg_monthly_spend,
    ROUND(AVG(spend_trend_3mo), 4)             AS avg_spend_trend_3mo,
    ROUND(AVG(tenure_months), 1)               AS avg_tenure_months
FROM card_customers
GROUP BY churn_flag;
 
 
-- 2.2 Churn rate by utilization ratio bucket (checks the U-shaped relationship)
SELECT
    CASE
        WHEN utilization_ratio < 0.2 THEN '0.0-0.2 (very low)'
        WHEN utilization_ratio < 0.4 THEN '0.2-0.4'
        WHEN utilization_ratio < 0.6 THEN '0.4-0.6'
        WHEN utilization_ratio < 0.8 THEN '0.6-0.8'
        ELSE '0.8-1.0 (very high)'
    END                                         AS utilization_bucket,
    COUNT(*)                                    AS customers,
    ROUND(AVG(churn_flag) * 100, 2)             AS churn_rate_pct
FROM card_customers
GROUP BY utilization_bucket
ORDER BY utilization_bucket;
 
 
-- 2.3 Churn rate by support call volume (engagement signal)
SELECT
    CASE
        WHEN support_calls_last_90d = 0 THEN '0 calls'
        WHEN support_calls_last_90d <= 2 THEN '1-2 calls'
        WHEN support_calls_last_90d <= 4 THEN '3-4 calls'
        ELSE '5+ calls'
    END                                         AS support_call_bucket,
    COUNT(*)                                    AS customers,
    ROUND(AVG(churn_flag) * 100, 2)             AS churn_rate_pct
FROM card_customers
GROUP BY support_call_bucket
ORDER BY MIN(support_calls_last_90d);
 
 
-- ============================================================
-- SECTION 3: RISK SEGMENTATION (SQL-native approximation of Phase 4 KMeans)
-- ============================================================
-- Note: the notebook's KMeans clustering is a Python-side ML step and
-- can't be replicated exactly in SQL. This section instead builds a
-- simple, transparent rule-based risk score using CASE/WHEN — a common
-- SQL-native alternative when a full ML pipeline isn't available at
-- the database layer, e.g. for a quick dashboard filter.
 
-- 3.1 Rule-based risk tier assignment and churn validation
WITH risk_scored AS (
    SELECT
        customer_id,
        churn_flag,
        (
            CASE WHEN tenure_months < 12 THEN 1 ELSE 0 END +
            CASE WHEN support_calls_last_90d >= 3 THEN 1 ELSE 0 END +
            CASE WHEN late_payments_last_12mo >= 2 THEN 1 ELSE 0 END +
            CASE WHEN spend_trend_3mo < -0.05 THEN 1 ELSE 0 END
        ) AS risk_points
    FROM card_customers
),
risk_tiered AS (
    SELECT
        customer_id,
        churn_flag,
        CASE
            WHEN risk_points >= 3 THEN 'high_risk'
            WHEN risk_points >= 1 THEN 'medium_risk'
            ELSE 'low_risk'
        END AS risk_tier
    FROM risk_scored
)
SELECT
    risk_tier,
    COUNT(*)                                    AS customers,
    ROUND(AVG(churn_flag) * 100, 2)             AS churn_rate_pct
FROM risk_tiered
GROUP BY risk_tier
ORDER BY churn_rate_pct;
 
 
-- ============================================================
-- SECTION 4: HIGH-RISK CUSTOMER TARGETING (mirrors Phase 5 output)
-- ============================================================
 
-- 4.1 Top 20 highest-risk-signal customers currently NOT churned
-- (candidates for a proactive save campaign — ranked by a simple
-- composite of known risk factors, since the true ML score lives in Python)
SELECT
    customer_id,
    cardholder_role,
    credit_score_band,
    tenure_months,
    ROUND(avg_monthly_spend, 2)                 AS avg_monthly_spend,
    spend_trend_3mo,
    support_calls_last_90d,
    late_payments_last_12mo
FROM card_customers
WHERE churn_flag = 0
ORDER BY
    (spend_trend_3mo) ASC,             -- most negative spend trend first
    late_payments_last_12mo DESC,
    support_calls_last_90d DESC
LIMIT 20;
 
 
-- ============================================================
-- SECTION 5: RETENTION OFFER / A-B TEST ANALYSIS (mirrors Phase 6)
-- ============================================================
 
-- 5.1 Naive retention rate by offer status
SELECT
    CASE WHEN received_offer = 1 THEN 'Offer' ELSE 'No Offer (Control)' END AS group_name,
    COUNT(*)                                    AS customers,
    SUM(retained_post_offer)                    AS retained,
    ROUND(AVG(retained_post_offer) * 100, 2)    AS retention_rate_pct
FROM offer_experiment
GROUP BY received_offer;
 
 
-- 5.2 Retention rate by specific offer type
SELECT
    offer_type,
    COUNT(*)                                    AS customers,
    ROUND(AVG(retained_post_offer) * 100, 2)    AS retention_rate_pct
FROM offer_experiment
GROUP BY offer_type
ORDER BY retention_rate_pct DESC;
 
 
-- 5.3 Offer effectiveness by credit score band (interaction effect)
SELECT
    credit_score_band,
    offer_type,
    COUNT(*)                                    AS customers,
    ROUND(AVG(retained_post_offer) * 100, 2)    AS retention_rate_pct
FROM offer_experiment
WHERE received_offer = 1
GROUP BY credit_score_band, offer_type
ORDER BY credit_score_band, retention_rate_pct DESC;
 
 
-- ============================================================
-- SECTION 6: COHORT RETENTION ANALYSIS (mirrors Phase 8)
-- ============================================================
-- Buckets customers into tenure-based cohorts and computes what
-- fraction are still retained at each tenure milestone -- the SQL
-- equivalent of the Python cohort retention curve.
 
-- 6.1 Retention rate at tenure milestones (1, 3, 6, 12 months)
SELECT
    'Month 1'  AS milestone,
    ROUND(AVG(CASE WHEN tenure_months >= 1  THEN (churn_flag = 0 OR tenure_months > 1)  ELSE NULL END) * 100, 2) AS retention_pct
FROM card_customers WHERE tenure_months >= 1
UNION ALL
SELECT
    'Month 3',
    ROUND(AVG(CASE WHEN tenure_months >= 3  THEN (churn_flag = 0 OR tenure_months > 3)  ELSE NULL END) * 100, 2)
FROM card_customers WHERE tenure_months >= 3
UNION ALL
SELECT
    'Month 6',
    ROUND(AVG(CASE WHEN tenure_months >= 6  THEN (churn_flag = 0 OR tenure_months > 6)  ELSE NULL END) * 100, 2)
FROM card_customers WHERE tenure_months >= 6
UNION ALL
SELECT
    'Month 12',
    ROUND(AVG(CASE WHEN tenure_months >= 12 THEN (churn_flag = 0 OR tenure_months > 12) ELSE NULL END) * 100, 2)
FROM card_customers WHERE tenure_months >= 12;
 
-- NOTE: this is a simplified point-in-time approximation (it treats "churned" customers as churned at their CURRENT tenure_months value, since this
-- snapshot dataset has no separate churn-event date). The Python notebook's Phase 8 simulates a churn month within the tenure window for a smoother curve. 
-- This query gives the SQL-native, more conservative version of the same idea, and is worth explaining as a limitation of snapshot-only data
-- in an interview setting.
 
 
-- 6.2 Signup cohort size and average tenure/churn by business size
-- (a business-size "cohort" view, since we lack true signup-date cohorts)
SELECT
    business_size,
    COUNT(*)                                    AS customers,
    ROUND(AVG(tenure_months), 1)                AS avg_tenure_months,
    ROUND(AVG(churn_flag) * 100, 2)             AS churn_rate_pct
FROM card_customers
GROUP BY business_size
ORDER BY avg_tenure_months DESC;
 
 
-- ============================================================
-- SECTION 7: EXECUTIVE SUMMARY QUERY
-- ============================================================
-- A single query producing the key headline metrics, suitable for
-- a dashboard summary card or a quick "state of the portfolio" check.
 
SELECT
    (SELECT COUNT(*) FROM card_customers)                                   AS total_customers,
    (SELECT ROUND(AVG(churn_flag) * 100, 2) FROM card_customers)            AS overall_churn_rate_pct,
    (SELECT ROUND(AVG(churn_flag) * 100, 2) FROM card_customers
     WHERE cardholder_role = 'business_owner')                              AS owner_churn_rate_pct,
    (SELECT ROUND(AVG(churn_flag) * 100, 2) FROM card_customers
     WHERE cardholder_role = 'employee')                                    AS employee_churn_rate_pct,
    (SELECT ROUND(AVG(retained_post_offer) * 100, 2) FROM offer_experiment
     WHERE received_offer = 1)                                              AS offer_retention_pct,
    (SELECT ROUND(AVG(retained_post_offer) * 100, 2) FROM offer_experiment
     WHERE received_offer = 0)                                              AS control_retention_pct;
 
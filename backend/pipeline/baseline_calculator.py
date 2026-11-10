"""
Baseline Deviation Calculator & Ground Truth Label Attacher
Computes:
- User's 30-day rolling baseline Z-score for all key telemetry features
- Department peer group Z-score using LDAP hierarchy
- Ground truth labels (is_malicious, scenario) from answers/insiders.csv
"""

import math
from datetime import timedelta
import pandas as pd
import numpy as np

METRICS = [
    'logon_count',
    'after_hours_logons',
    'usb_connections',
    'files_copied',
    'emails_sent',
    'external_emails',
    'total_websites_visited',
    'suspicious_url_visits'
]


def compute_baselines_and_labels(daily_records, user_metadata, malicious_intervals):
    """
    Takes daily records dictionary, enriches with LDAP metadata,
    computes rolling 30-day individual Z-scores, department peer Z-scores,
    and ground truth labels.
    Returns a pandas DataFrame.
    """
    print("\n[*] Enriching records with LDAP metadata and ground-truth labels...")
    rows = list(daily_records.values())
    if not rows:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    df.sort_values(by=['user', 'date_obj'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1. Attach LDAP attributes
    df['department'] = df['user'].map(lambda u: user_metadata.get(u, {}).get('department', 'Unknown'))
    df['role'] = df['user'].map(lambda u: user_metadata.get(u, {}).get('role', 'Unknown'))
    df['functional_unit'] = df['user'].map(lambda u: user_metadata.get(u, {}).get('functional_unit', 'Unknown'))

    # 2. Attach ground truth labels (only 1 for the days the insider was actually malicious)
    labels = []
    scenarios = []
    for _, row in df.iterrows():
        u = row['user']
        d = row['date_obj']
        intervals = malicious_intervals.get(u, [])
        is_mal = 0
        scen = 0
        for inv in intervals:
            if inv['start_date'] <= d <= inv['end_date']:
                is_mal = 1
                scen = inv['scenario']
                break
        labels.append(is_mal)
        scenarios.append(scen)

    df['is_malicious'] = labels
    df['malicious_scenario'] = scenarios

    # 3. Compute rolling 30-day individual baseline Z-scores
    print("[*] Computing 30-day rolling individual baselines...")
    for m in METRICS:
        df[f'z_own_30d_{m}'] = 0.0

    # Process each user
    grouped_users = df.groupby('user')
    z_own_series = {f'z_own_30d_{m}': [] for m in METRICS}

    for user, u_df in grouped_users:
        u_records = u_df[['date_obj'] + METRICS].to_dict('records')
        n = len(u_records)
        
        # Maintain history of prior 30 days
        for i in range(n):
            curr_date = u_records[i]['date_obj']
            window_start = curr_date - timedelta(days=30)
            
            # Prior records strictly before curr_date and >= window_start
            prior_vals = [
                u_records[j] for j in range(i)
                if u_records[j]['date_obj'] >= window_start
            ]
            
            for m in METRICS:
                curr_val = u_records[i][m]
                if len(prior_vals) >= 2:
                    vals = [p[m] for p in prior_vals]
                    mean = sum(vals) / len(vals)
                    var = sum((v - mean)**2 for v in vals) / (len(vals) - 1)
                    sd = math.sqrt(var)
                    z = (curr_val - mean) / (sd + 1.0)
                elif len(prior_vals) == 1:
                    mean = prior_vals[0][m]
                    z = (curr_val - mean) / (1.0 + abs(mean * 0.2))
                else:
                    z = 0.0
                z_own_series[f'z_own_30d_{m}'].append(round(z, 3))

    for m in METRICS:
        df[f'z_own_30d_{m}'] = z_own_series[f'z_own_30d_{m}']

    # 4. Compute Department Peer Group Z-scores
    print("[*] Computing department peer group Z-scores...")
    dept_stats = {}
    for dept, d_df in df.groupby('department'):
        dept_stats[dept] = {}
        for m in METRICS:
            m_vals = d_df[m].values
            mean = float(np.mean(m_vals))
            sd = float(np.std(m_vals, ddof=1)) if len(m_vals) > 1 else 1.0
            dept_stats[dept][m] = (mean, max(sd, 0.5))

    for m in METRICS:
        z_col = f'z_dept_{m}'
        df[z_col] = df.apply(
            lambda r: round((r[m] - dept_stats.get(r['department'], {}).get(m, (0, 1))[0]) / 
                            (dept_stats.get(r['department'], {}).get(m, (0, 1))[1] + 1.0), 3),
            axis=1
        )

    # Convert date_obj back to string for parquet serialization
    df['date'] = df['date_obj'].astype(str)
    df.drop(columns=['date_obj'], inplace=True)

    print(f"[*] Baseline & label calculations complete. Final shape: {df.shape}")
    return df

"""
Main Execution Script for CERT r4.2 Data Pipeline
Extracts daily features, computes dual baselines, and generates features.parquet
"""

import os
import sys
import time

# Ensure pipeline package is importable
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(PIPELINE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from pipeline.config import PROCESSED_DATA_DIR
from pipeline.user_selector import get_user_subset
from pipeline.ldap_loader import load_ldap_metadata
from pipeline.feature_extractor import extract_daily_telemetry
from pipeline.baseline_calculator import compute_baselines_and_labels


def run():
    print("=" * 70)
    print("  CERT r4.2 DATA PIPELINE: MULTI-VECTOR FEATURE GENERATION")
    print("=" * 70)
    start_time = time.time()

    # 1. Select Users
    user_info = get_user_subset()
    selected_users = user_info['selected_users']
    malicious_users = user_info['malicious_users']
    malicious_intervals = user_info['malicious_intervals']

    # 2. Load LDAP Metadata
    user_metadata = load_ldap_metadata()

    # 3. Stream & Aggregate Daily Telemetry
    daily_records = extract_daily_telemetry(selected_users)

    # 4. Compute 30-Day & Department Baselines + Labels
    df = compute_baselines_and_labels(daily_records, user_metadata, malicious_intervals)

    if df.empty:
        print("[!] Error: No features extracted.")
        return None

    # 5. Save Output
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    parquet_path = os.path.join(PROCESSED_DATA_DIR, "features.parquet")
    csv_path = os.path.join(PROCESSED_DATA_DIR, "features.csv")

    print(f"\n[*] Saving feature table to: {parquet_path}")
    df.to_parquet(parquet_path, index=False, engine='pyarrow')

    print(f"[*] Saving CSV copy to: {csv_path}")
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n[+] Pipeline finished successfully in {elapsed:.1f}s!")

    # 6. Summary Statistics
    total_rows = len(df)
    unique_users = df['user'].nunique()
    mal_users_present = df[df['is_malicious'] == 1]['user'].nunique()
    mal_days = (df['is_malicious'] == 1).sum()
    normal_days = (df['is_malicious'] == 0).sum()
    pct_mal = (mal_days / total_rows) * 100

    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY & CLASS BALANCE:")
    print("=" * 70)
    print(f"- Total rows (user-days): {total_rows:,}")
    print(f"- Total unique users: {unique_users} (70 malicious + 100 normal sample)")
    print(f"- Malicious users with active incident days: {mal_users_present}")
    print(f"- Normal days (Class 0): {normal_days:,} ({(normal_days/total_rows)*100:.2f}%)")
    print(f"- Malicious days (Class 1): {mal_days:,} ({pct_mal:.2f}%)")
    print(f"- Malicious scenario breakdown:")
    for scen, count in df[df['is_malicious'] == 1]['malicious_scenario'].value_counts().items():
        print(f"    Scenario {scen}: {count} malicious days")

    print("\n5-ROW PREVIEW (Sample with both normal & malicious records):")
    sample_preview = df[df['is_malicious'] == 1].head(3)
    sample_normal = df[df['is_malicious'] == 0].head(2)
    preview = pd.concat([sample_preview, sample_normal])
    
    preview_cols = [
        'user', 'date', 'department', 'role',
        'logon_count', 'after_hours_logons', 'usb_connections',
        'files_copied', 'emails_sent', 'external_emails',
        'total_websites_visited', 'suspicious_url_visits',
        'z_own_30d_usb_connections', 'z_own_30d_files_copied',
        'is_malicious', 'malicious_scenario'
    ]
    print(preview[preview_cols].to_string())

    return df


if __name__ == "__main__":
    run()

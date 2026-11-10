"""
User Selector & Malicious Ground-Truth Label Mapper
"""

import os
import csv
import random
from datetime import datetime
from .config import RAW_DATA_DIR, ANSWERS_DIR, RANDOM_SEED, NORMAL_USER_SAMPLE_SIZE


def parse_date_only(date_str):
    """
    Parses 'MM/DD/YYYY HH:MM:SS' or 'M/D/YYYY H:M:S' into a date object.
    """
    date_part = date_str.strip().split()[0]
    parts = [int(p) for p in date_part.split('/')]
    month, day, year = parts[0], parts[1], parts[2]
    return datetime(year, month, day).date()


def get_user_subset():
    """
    Extracts all 70 malicious users for r4.2 plus a deterministic sample of
    100 normal users (total 170 users).
    """
    insiders_file = os.path.join(ANSWERS_DIR, "insiders.csv")
    malicious_intervals = {}  # user -> list of {start_date, end_date, scenario}
    malicious_users = set()

    with open(insiders_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('dataset') == '4.2':
                u = row['user'].strip()
                s_date = parse_date_only(row['start'])
                e_date = parse_date_only(row['end'])
                scen = int(row['scenario'])

                malicious_users.add(u)
                malicious_intervals.setdefault(u, []).append({
                    'start_date': s_date,
                    'end_date': e_date,
                    'scenario': scen,
                    'details': row.get('details', '')
                })

    # Read all users from psychometric.csv
    psych_file = os.path.join(RAW_DATA_DIR, "psychometric.csv")
    all_users = set()
    with open(psych_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_users.add(row['user_id'].strip())

    normal_users = sorted(all_users - malicious_users)
    rng = random.Random(RANDOM_SEED)
    sampled_normal_users = set(rng.sample(normal_users, min(NORMAL_USER_SAMPLE_SIZE, len(normal_users))))

    selected_users = malicious_users | sampled_normal_users

    print(f"[*] Total dataset users: {len(all_users)}")
    print(f"[*] Malicious users (100% included): {len(malicious_users)}")
    print(f"[*] Normal users sampled: {len(sampled_normal_users)} (Seed={RANDOM_SEED})")
    print(f"[*] Target user subset size: {len(selected_users)} users")

    return {
        'selected_users': selected_users,
        'malicious_users': malicious_users,
        'sampled_normal_users': sampled_normal_users,
        'malicious_intervals': malicious_intervals
    }


def is_malicious_day(user, date_obj, malicious_intervals):
    """
    Returns (1, scenario) if user was actively malicious on date_obj, else (0, 0).
    """
    intervals = malicious_intervals.get(user)
    if not intervals:
        return 0, 0

    for inv in intervals:
        if inv['start_date'] <= date_obj <= inv['end_date']:
            return 1, inv['scenario']

    return 0, 0

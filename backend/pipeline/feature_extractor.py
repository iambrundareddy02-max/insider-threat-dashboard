"""
Chunked Streaming Telemetry Extractor for CERT r4.2
Aggregates activity into daily summaries per user.
"""

import os
import csv
import time
from datetime import datetime
from .config import (
    RAW_DATA_DIR,
    JOB_SEARCH_KEYWORDS,
    CLOUD_STORAGE_KEYWORDS,
    LEAKER_KEYWORDS,
    COMPANY_EMAIL_DOMAIN
)


def parse_datetime_parts(raw_date_str):
    """
    Parses 'MM/DD/YYYY HH:MM:SS' into (norm_date_str, date_obj, hour, is_after_hours, is_weekend)
    """
    try:
        d_part, t_part = raw_date_str.strip().split()
        m_s, d_s, y_s = d_part.split('/')
        m, d, y = int(m_s), int(d_s), int(y_s)
        norm_date = f"{y:04d}-{m:02d}-{d:02d}"
        date_obj = datetime(y, m, d).date()
        hour = int(t_part.split(':')[0])
        weekday = date_obj.weekday()
        after_hours = 1 if (hour < 8 or hour >= 18) else 0
        weekend = 1 if (weekday >= 5) else 0
        return norm_date, date_obj, hour, after_hours, weekend
    except Exception:
        return None, None, 0, 0, 0


def create_empty_record(user, date_str, date_obj):
    return {
        'user': user,
        'date': date_str,
        'date_obj': date_obj,
        'logon_count': 0,
        'after_hours_logons': 0,
        'weekend_logons': 0,
        'usb_connections': 0,
        'files_copied': 0,
        'emails_sent': 0,
        'external_emails': 0,
        'attachment_size': 0,
        'total_websites_visited': 0,
        'suspicious_url_visits': 0,
        'job_search_visits': 0,
        'cloud_storage_visits': 0,
        'leaker_visits': 0
    }


def extract_daily_telemetry(selected_users):
    """
    Streams through all CERT raw logs in chunks, filtering strictly by selected_users,
    and returns a mapping: (user, date_str) -> daily feature record.
    """
    daily_records = {}
    total_start = time.time()

    def get_rec(u, d_str, d_obj):
        key = (u, d_str)
        if key not in daily_records:
            daily_records[key] = create_empty_record(u, d_str, d_obj)
        return daily_records[key]

    # -------------------------------------------------------------
    # 1. LOGON.CSV
    # -------------------------------------------------------------
    logon_path = os.path.join(RAW_DATA_DIR, "logon.csv")
    print(f"\n[1/5] Processing logon.csv for {len(selected_users)} target users...")
    t0 = time.time()
    count_matched = 0
    with open(logon_path, 'r', encoding='utf-8', errors='replace', buffering=8*1024*1024) as f:
        reader = csv.reader(f)
        header = next(reader)
        # Expected: id, date, user, pc, activity
        for row in reader:
            if len(row) < 5:
                continue
            user = row[2]
            if user not in selected_users:
                continue

            count_matched += 1
            raw_date = row[1]
            norm_date, d_obj, hour, after_h, weekend = parse_datetime_parts(raw_date)
            if not norm_date:
                continue

            rec = get_rec(user, norm_date, d_obj)
            activity = row[4].strip()
            if activity == 'Logon':
                rec['logon_count'] += 1
                rec['after_hours_logons'] += after_h
                rec['weekend_logons'] += weekend

    print(f"      Matched {count_matched:,} logon events in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------
    # 2. DEVICE.CSV
    # -------------------------------------------------------------
    device_path = os.path.join(RAW_DATA_DIR, "device.csv")
    print(f"\n[2/5] Processing device.csv...")
    t0 = time.time()
    count_matched = 0
    with open(device_path, 'r', encoding='utf-8', errors='replace', buffering=8*1024*1024) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 5:
                continue
            user = row[2]
            if user not in selected_users:
                continue

            count_matched += 1
            raw_date = row[1]
            norm_date, d_obj, hour, after_h, weekend = parse_datetime_parts(raw_date)
            if not norm_date:
                continue

            rec = get_rec(user, norm_date, d_obj)
            activity = row[4].strip()
            if activity == 'Connect':
                rec['usb_connections'] += 1

    print(f"      Matched {count_matched:,} device events in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------
    # 3. FILE.CSV
    # -------------------------------------------------------------
    file_path = os.path.join(RAW_DATA_DIR, "file.csv")
    print(f"\n[3/5] Processing file.csv...")
    t0 = time.time()
    count_matched = 0
    with open(file_path, 'r', encoding='utf-8', errors='replace', buffering=8*1024*1024) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            user = row[2]
            if user not in selected_users:
                continue

            count_matched += 1
            raw_date = row[1]
            norm_date, d_obj, hour, after_h, weekend = parse_datetime_parts(raw_date)
            if not norm_date:
                continue

            rec = get_rec(user, norm_date, d_obj)
            rec['files_copied'] += 1

    print(f"      Matched {count_matched:,} file copy events in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------
    # 4. EMAIL.CSV
    # -------------------------------------------------------------
    email_path = os.path.join(RAW_DATA_DIR, "email.csv")
    print(f"\n[4/5] Processing email.csv...")
    t0 = time.time()
    count_matched = 0
    with open(email_path, 'r', encoding='utf-8', errors='replace', buffering=16*1024*1024) as f:
        reader = csv.reader(f)
        header = next(reader)
        # Expected: id, date, user, pc, to, cc, bcc, from, size, attachments, content
        for row in reader:
            if len(row) < 10:
                continue
            user = row[2]
            if user not in selected_users:
                continue

            count_matched += 1
            raw_date = row[1]
            norm_date, d_obj, hour, after_h, weekend = parse_datetime_parts(raw_date)
            if not norm_date:
                continue

            rec = get_rec(user, norm_date, d_obj)
            rec['emails_sent'] += 1

            # Check external recipients
            recipients = f"{row[4]};{row[5]};{row[6]}".lower()
            if recipients:
                for addr in recipients.split(';'):
                    addr = addr.strip()
                    if '@' in addr and not addr.endswith(f"@{COMPANY_EMAIL_DOMAIN}"):
                        rec['external_emails'] += 1
                        break

            # Attachment size
            attachments_str = row[9].strip()
            if attachments_str.isdigit() and int(attachments_str) > 0:
                size_str = row[8].strip()
                if size_str.isdigit():
                    rec['attachment_size'] += int(size_str)

    print(f"      Matched {count_matched:,} email events in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------
    # 5. HTTP.CSV (FAST STREAMING)
    # -------------------------------------------------------------
    http_path = os.path.join(RAW_DATA_DIR, "http.csv")
    print(f"\n[5/5] Processing http.csv (28.4M lines streamed in chunks)...")
    t0 = time.time()
    count_matched = 0
    total_lines = 0

    with open(http_path, 'r', encoding='utf-8', errors='replace', buffering=32*1024*1024) as f:
        f.readline()  # Skip header
        for line in f:
            total_lines += 1
            parts = line.split(',', 5)
            if len(parts) < 5:
                continue
            user = parts[2]
            if user not in selected_users:
                continue

            count_matched += 1
            raw_date = parts[1]
            norm_date, d_obj, hour, after_h, weekend = parse_datetime_parts(raw_date)
            if not norm_date:
                continue

            rec = get_rec(user, norm_date, d_obj)
            rec['total_websites_visited'] += 1

            url = parts[4].lower()
            is_job = any(k in url for k in JOB_SEARCH_KEYWORDS)
            is_cloud = any(k in url for k in CLOUD_STORAGE_KEYWORDS)
            is_leaker = any(k in url for k in LEAKER_KEYWORDS)

            if is_job:
                rec['job_search_visits'] += 1
            if is_cloud:
                rec['cloud_storage_visits'] += 1
            if is_leaker:
                rec['leaker_visits'] += 1
            if is_job or is_cloud or is_leaker:
                rec['suspicious_url_visits'] += 1

    print(f"      Matched {count_matched:,} HTTP events across {total_lines:,} lines in {time.time()-t0:.1f}s")
    print(f"[*] Total unique user-day records generated: {len(daily_records):,} in {time.time()-total_start:.1f}s total.")

    return daily_records

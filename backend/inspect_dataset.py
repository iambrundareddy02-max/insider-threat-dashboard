"""
Phase 1: CERT r4.2 Dataset & Answers Inspector
Inspects file sizes, column schemas, row counts, date ranges, and insiders.csv.
Uses streaming chunked readers for massive CSVs.
"""

import os
import glob
import time
import csv
from datetime import datetime

DATA_DIR = r"C:\Users\manik\.gemini\antigravity\scratch\insider-threat-web\backend\data\raw\r4.2"
ANSWERS_DIR = r"C:\Users\manik\.gemini\antigravity\scratch\insider-threat-web\backend\data\raw\answers"

def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def inspect_csv_streaming(filepath, date_col_name="date", chunk_size=1024*1024):
    """
    Counts rows and tracks date range without loading everything into RAM.
    """
    file_size = os.path.getsize(filepath)
    print(f"\n[*] Inspecting {os.path.basename(filepath)} ({format_size(file_size)})...")
    start_time = time.time()
    
    columns = []
    row_count = 0
    min_date = None
    max_date = None
    corrupt_rows = 0

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        try:
            columns = next(reader)
        except StopIteration:
            return {"file": filepath, "size": file_size, "rows": 0, "cols": [], "error": "Empty file"}

        date_idx = None
        if date_col_name:
            for idx, col in enumerate(columns):
                if col.lower() == date_col_name.lower():
                    date_idx = idx
                    break

        # Fast line-based iteration or chunked csv parsing
        # For very large files, parse sample for dates and line-count
        sample_rows_for_dates = 50000

        for row in reader:
            row_count += 1
            if len(row) != len(columns):
                corrupt_rows += 1
                continue
            
            # Check dates for first N rows and periodically
            if date_idx is not None and (row_count <= sample_rows_for_dates or row_count % 1000 == 0):
                d_str = row[date_idx]
                if d_str:
                    try:
                        # Common CERT formats: 'MM/DD/YYYY HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS'
                        if min_date is None or d_str < min_date:
                            min_date = d_str
                        if max_date is None or d_str > max_date:
                            max_date = d_str
                    except Exception:
                        pass

    elapsed = time.time() - start_time
    print(f"    Done in {elapsed:.1f}s: {row_count:,} rows, cols: {columns}")
    return {
        "file": os.path.basename(filepath),
        "size_bytes": file_size,
        "size_formatted": format_size(file_size),
        "columns": columns,
        "row_count": row_count,
        "min_date": min_date,
        "max_date": max_date,
        "corrupt_rows": corrupt_rows,
        "elapsed_sec": round(elapsed, 2)
    }

def inspect_http_fast(filepath):
    """
    http.csv is ~14.5 GB. We read header, inspect columns and sample rows,
    then quickly count newlines using raw binary chunk reads.
    """
    file_size = os.path.getsize(filepath)
    print(f"\n[*] Inspecting http.csv ({format_size(file_size)}) via binary buffered streaming...")
    start_time = time.time()

    columns = []
    min_date = None
    max_date = None
    date_idx = None

    # 1. Read header and first 10,000 rows for schema and date range sample
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        columns = next(reader)
        for idx, col in enumerate(columns):
            if col.lower() == 'date':
                date_idx = idx
                break
        
        for i in range(10000):
            try:
                row = next(reader)
                if date_idx is not None and len(row) > date_idx:
                    d_str = row[date_idx]
                    if min_date is None or d_str < min_date:
                        min_date = d_str
                    if max_date is None or d_str > max_date:
                        max_date = d_str
            except StopIteration:
                break

    # 2. Fast line count using binary chunk read
    line_count = 0
    chunk_size = 32 * 1024 * 1024  # 32MB chunks
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            line_count += chunk.count(b'\n')

    # Subtract header
    row_count = max(0, line_count - 1)
    elapsed = time.time() - start_time
    print(f"    Done http.csv in {elapsed:.1f}s: ~{row_count:,} rows")

    return {
        "file": os.path.basename(filepath),
        "size_bytes": file_size,
        "size_formatted": format_size(file_size),
        "columns": columns,
        "row_count": row_count,
        "min_date": min_date,
        "max_date": max_date,
        "corrupt_rows": 0,
        "elapsed_sec": round(elapsed, 2)
    }

def inspect_ldap(ldap_dir):
    files = sorted(glob.glob(os.path.join(ldap_dir, "*.csv")))
    total_size = sum(os.path.getsize(f) for f in files)
    print(f"\n[*] Inspecting LDAP folder ({len(files)} files, total {format_size(total_size)})...")
    
    sample_cols = []
    total_rows = 0
    dates = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        dates.append(fname.replace(".csv", ""))
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader)
            if not sample_cols:
                sample_cols = header
            rows = sum(1 for _ in reader)
            total_rows += rows

    return {
        "folder": "LDAP",
        "file_count": len(files),
        "files": [os.path.basename(f) for f in files],
        "size_bytes": total_size,
        "size_formatted": format_size(total_size),
        "columns": sample_cols,
        "row_count": total_rows,
        "date_range": f"{dates[0]} to {dates[-1]} ({len(files)} months)"
    }

def inspect_insiders(insiders_path):
    print(f"\n[*] Inspecting answers/insiders.csv...")
    insiders = []
    with open(insiders_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('dataset') == '4.2':
                insiders.append(row)
    return insiders

if __name__ == "__main__":
    results = {}
    
    # 1. Inspect logon.csv
    results["logon"] = inspect_csv_streaming(os.path.join(DATA_DIR, "logon.csv"))
    
    # 2. Inspect device.csv
    results["device"] = inspect_csv_streaming(os.path.join(DATA_DIR, "device.csv"))
    
    # 3. Inspect file.csv
    results["file"] = inspect_csv_streaming(os.path.join(DATA_DIR, "file.csv"))
    
    # 4. Inspect psychometric.csv
    results["psychometric"] = inspect_csv_streaming(os.path.join(DATA_DIR, "psychometric.csv"), date_col_name=None)
    
    # 5. Inspect LDAP
    results["ldap"] = inspect_ldap(os.path.join(DATA_DIR, "LDAP"))
    
    # 6. Inspect email.csv (chunked)
    results["email"] = inspect_csv_streaming(os.path.join(DATA_DIR, "email.csv"))

    # 7. Inspect http.csv (fast binary count)
    results["http"] = inspect_http_fast(os.path.join(DATA_DIR, "http.csv"))
    
    # 8. Inspect insiders.csv
    insiders_42 = inspect_insiders(os.path.join(ANSWERS_DIR, "insiders.csv"))
    
    print("\n" + "="*70)
    print("PHASE 1 SUMMARY RESULTS:")
    print("="*70)
    for k, v in results.items():
        if k == "ldap":
            print(f"- LDAP: {v['file_count']} files, {v['size_formatted']}, {v['row_count']:,} total employee-month entries, range: {v['date_range']}")
            print(f"  Columns: {v['columns']}")
        else:
            print(f"- {v['file']}: {v['size_formatted']}, {v['row_count']:,} rows, min date: {v['min_date']}, max date: {v['max_date']}")
            print(f"  Columns: {v['columns']}")

    print(f"\n- answers/insiders.csv (r4.2): {len(insiders_42)} malicious insider scenarios found.")
    for ins in insiders_42:
        print(f"  Scenario {ins['scenario']}: User {ins['user']} from {ins['start']} to {ins['end']} (Details: {ins['details']})")

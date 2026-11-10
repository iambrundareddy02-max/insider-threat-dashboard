"""
LDAP Organizational Hierarchy & Department Metadata Loader
"""

import os
import glob
import csv
from .config import RAW_DATA_DIR


def load_ldap_metadata():
    """
    Parses monthly LDAP snapshot CSVs and maps each user_id to their
    organizational department, role, functional unit, and supervisor.
    """
    ldap_dir = os.path.join(RAW_DATA_DIR, "LDAP")
    files = sorted(glob.glob(os.path.join(ldap_dir, "*.csv")))
    
    user_metadata = {}
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row['user_id'].strip()
                # Store latest or primary metadata
                user_metadata[uid] = {
                    'employee_name': row.get('employee_name', ''),
                    'role': row.get('role', 'Unknown'),
                    'department': row.get('department', 'Unassigned'),
                    'functional_unit': row.get('functional_unit', ''),
                    'business_unit': row.get('business_unit', ''),
                    'supervisor': row.get('supervisor', '')
                }

    print(f"[*] Loaded LDAP organizational metadata for {len(user_metadata)} employees across {len(files)} snapshots.")
    return user_metadata

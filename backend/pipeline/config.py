"""
Pipeline Configuration & Suspicious Domain Mappings
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "r4.2")
ANSWERS_DIR = os.path.join(BASE_DIR, "data", "raw", "answers")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# URL Keywords to classify suspicious web visits
JOB_SEARCH_KEYWORDS = [
    'monster.com', 'indeed.com', 'careerbuilder.com', 'dice.com',
    'simplyhired.com', 'linkedin.com/jobs', 'jobsearch', 'resume',
    'snagajob.com', 'career', 'employment', 'jobs.com'
]

CLOUD_STORAGE_KEYWORDS = [
    'dropbox.com', 'drive.google.com', 'box.com', 'onedrive.live.com',
    'mega.co.nz', 'filedropper.com', 'sendspace.com', 'rapidshare.com',
    'mediafire.com', '4shared.com', 'depositfiles.com', 'turbobit.net'
]

LEAKER_KEYWORDS = [
    'wikileaks.org', 'pastebin.com', 'cryptome.org', 'torrent',
    'thepiratebay', 'leaks', 'btdigg.org', 'anonpaste'
]

# Random seed for reproducible normal user sampling
RANDOM_SEED = 42
NORMAL_USER_SAMPLE_SIZE = 100
COMPANY_EMAIL_DOMAIN = "dtaa.com"

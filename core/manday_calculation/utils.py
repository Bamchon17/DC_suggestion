import sys
import os
import re
import difflib
import pandas as pd
import numpy as np
from collections import defaultdict
import google.generativeai as genai

# -------------------- Path --------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
from db.connection import conn  # ใช้การเชื่อมต่อจาก connection.py

# -------------------- Gemini --------------------
API_KEY = os.getenv("gemini_api_key")
if not API_KEY:
    raise ValueError("gemini_api_key not found in .env file")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------- Utilities --------------------
TH_SEP_PATTERN = re.compile(r"\s*(/|\\|\||;|และ|,|，|、|：|:|\+|\s+และ\s+)\s*")
NON_ALNUM_TH = re.compile(r"[^0-9a-zA-Zก-๙\s,]")
MULTI_SPACE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.strip()
    s = NON_ALNUM_TH.sub(" ", s)  # drop weird punctuation but keep commas
    s = MULTI_SPACE.sub(" ", s)
    return s

def normalize_skill_format(s: str) -> str:
    """Return comma-separated skills w/ single spaces, no slashes."""
    s = normalize_text(s).lower()
    if not s:
        return s
    s2 = TH_SEP_PATTERN.sub(",", s)
    items = [itm.strip() for itm in s2.split(",") if itm and itm.strip()]
    seen = set()
    uniq = []
    for itm in items:
        if itm not in seen:
            seen.add(itm)
            uniq.append(itm)
    return ", ".join(uniq)

def best_string_match(term: str, choices: list[str], cutoff_exact: float = 0.999, cutoff_close: float = 0.8) -> str | None:
    if not term:
        return None
    t = term.strip().lower()
    if not choices:
        return None
    low = [c.strip().lower() for c in choices]
    if t in low:
        return choices[low.index(t)]
    for i, l in enumerate(low):
        if t in l and len(t) >= 3:
            return choices[i]
    for i, l in enumerate(low):
        if l in t and len(l) >= 3:
            return choices[i]
    matches = difflib.get_close_matches(t, low, n=1, cutoff=cutoff_close)
    if matches:
        return choices[low.index(matches[0])]
    return None
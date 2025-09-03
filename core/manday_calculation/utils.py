# core/manday_calculation/utils.py
import re
import pandas as pd
import google.generativeai as genai

# --- Gemini model ---
model = genai.GenerativeModel()

# --- Text normalization ---
TH_SEP_PATTERN = re.compile(r'(;และ|，|、|：|\s+และ\s+)')
NON_ALNUM_TH = re.compile(r'[^0-9a-zA-Zก-๙\s,]')
MULTI_SPACE = re.compile(r'\s+')
INT_OR_FLOAT = re.compile(r'(\d+(\.\d+)?)')

def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ''
    s = str(s).strip()
    s = NON_ALNUM_TH.sub(' ', s)
    s = MULTI_SPACE.sub(' ', s)
    return s

def normalize_skill_format(s: str) -> str:
    s = normalize_text(s).lower()
    if not s:
        return s
    s2 = TH_SEP_PATTERN.sub(',', s)
    items = [itm.strip() for itm in s2.split(',') if itm and itm.strip()]
    seen = set()
    uniq = []
    for itm in items:
        if itm not in seen:
            seen.add(itm)
            uniq.append(itm)
    return ', '.join(uniq)

def best_string_match(term: str, choices: list[str], cutoff_exact: float = 0.999, cutoff_close: float = 0.6) -> str | None:
    import difflib
    if not term or not choices:
        return None
    t = term.strip().lower()
    low = [c.strip().lower() for c in choices]
    if t in low:
        return choices[low.index(t)]
    for i, l in enumerate(low):
        if t in l and len(t) == 3:
            return choices[i]
        if l in t and len(l) == 3:
            return choices[i]
    matches = difflib.get_close_matches(t, low, n=1, cutoff=cutoff_close)
    if matches:
        return choices[low.index(matches[0])]
    return None

def parse_skill_level(val):
    text = str(val).strip() if not pd.isna(val) else ''
    if re.search(r'ทดลองฝึกงาน|intern', text, re.IGNORECASE):
        return 0.0, 'U'
    m = INT_OR_FLOAT.search(text)
    num = float(m.group(1)) if m else None
    suffix = ''
    if m:
        after = text[m.end():].strip()
        sm = re.match(r'([A-Za-z]+)', after)
        suffix = sm.group(1) if sm else ''
    return num, suffix

def clean_skill_level(df_skills: pd.DataFrame) -> pd.DataFrame:
    parsed = df_skills['skill_level'].apply(parse_skill_level)
    df_skills = df_skills.copy()
    df_skills['skill_level_num'] = parsed.apply(lambda x: x[0])
    df_skills['skill_level_suffix'] = parsed.apply(lambda x: x[1])
    return df_skills[df_skills['skill_level_num'].notna()]

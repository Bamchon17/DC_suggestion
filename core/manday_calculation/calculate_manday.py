import sys
import os
import re
import difflib
import pandas as pd
import numpy as np
import psycopg2
from collections import defaultdict
from dotenv import load_dotenv
import google.generativeai as genai
from sqlalchemy import create_engine, text
import uuid

# -------------------- Path & ENV --------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
from db.connection import conn  # expects a `conn` psycopg2 connection

load_dotenv("db/.env")

# ดึงค่าจาก .env
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# สร้าง Connection URL สำหรับ Postgres (Neon ใช้ sslmode=require)
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"

# สร้าง engine
engine = create_engine(DATABASE_URL)

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

# -------------------- DB Helpers --------------------
def fetch_query(conn, query, columns=None):
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        return pd.DataFrame(rows, columns=columns) if columns else pd.DataFrame(rows)
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

# -------------------- Queries --------------------
worker_query = """
SELECT  
    w.worker_id,
    w.worker_name,
    st.skill_name,
    sr.skill_level,
    sr.evaluation
FROM worker AS w
JOIN skill_record AS sr ON w.worker_id = sr.worker_id
JOIN skill_type AS st ON st.skill_type_id = sr.skill_type_id
"""

project_query = """
SELECT pr.project_id, 
       pr.project_name, 
       t.task_id, t.task_name,
       st.subtask_id, 
       st.sub_task_name, 
       st.qty, st.unit, 
       st.start_date, 
       st.end_date,
       st.durations_subtask
FROM projects AS pr
JOIN tasks AS t ON t.project_id = pr.project_id
JOIN subtask AS st ON st.task_id = t.task_id
"""

work_standard_query = """
SELECT task_name, standard_rate, unit
FROM work_standard
"""

# -------------------- Skill level parsing --------------------
INT_OR_FLOAT = re.compile(r"(\d+(?:\.\d+)?)")

def parse_skill_level(val):
    text = str(val).strip() if not pd.isna(val) else ""
    if re.search(r"ทดลอง|ฝึกงาน|intern", text, re.IGNORECASE):
        return 0.0, "U"
    m = INT_OR_FLOAT.search(text)
    num = float(m.group(1)) if m else None
    suffix = ""
    if m:
        after = text[m.end():].strip()
        sm = re.match(r"([A-Za-z]+)", after)
        suffix = sm.group(1) if sm else ""
    return num, suffix

def clean_skill_level(df_skills: pd.DataFrame) -> pd.DataFrame:
    parsed = df_skills["skill_level"].apply(parse_skill_level)
    df_skills = df_skills.copy()
    df_skills["skill_level_num"] = parsed.apply(lambda x: x[0])
    df_skills["skill_level_suffix"] = parsed.apply(lambda x: x[1])
    return df_skills[df_skills["skill_level_num"].notna()]

# -------------------- Gemini Matching --------------------
def match_skills_to_tasks(worker_skills_df: pd.DataFrame, project_df: pd.DataFrame, model: genai.GenerativeModel) -> pd.DataFrame:
    skills_pool = (
        worker_skills_df["skill_name"].dropna().astype(str).map(normalize_text).str.lower().unique().tolist()
    )
    if not skills_pool:
        print("⚠️ No skills found in worker_skills_df.")
        return pd.DataFrame()

    task_lines = [
        f"- งาน: {normalize_text(row['task_name'])}, รายละเอียดงานย่อย: {normalize_text(row['sub_task_name'])}, task_id: {row['task_id']}, subtask_id: {row['subtask_id']}"
        for _, row in project_df.iterrows()
    ]
    if not task_lines:
        print("⚠️ No tasks found in project_df.")
        return pd.DataFrame()

    prompt = f"""
คุณเป็นผู้ช่วยจับคู่ทักษะแรงงานกับงานก่อสร้าง ให้เลือกเพียงชื่อทักษะหลักจากรายการต่อไปนี้เท่านั้น:
{', '.join(skills_pool)}

ข้อกำหนดรูปแบบคำตอบ (สำคัญมาก):
- ส่งข้อมูลเป็นตาราง Markdown 5 คอลัมน์ (ชื่องาน | งานย่อย | ทักษะที่จับคู่ได้ | task_id | subtask_id)
- ถ้ามีมากกว่า 1 ทักษะ ให้คั่นด้วยเครื่องหมายจุลภาคและช่องว่าง เช่น "ช่างฝ้า, งานฝ้า" ห้ามใช้เครื่องหมาย "/" ใด ๆ
- ถ้าจับคู่ไม่ได้ ให้ใส่คำว่า "ไม่สามารถจับคู่ได้"
- คงค่า task_id และ subtask_id จากข้อมูลที่ให้มา

รายการงานที่ต้องการจับคู่ทักษะ:
{chr(10).join(task_lines)}
"""

    try:
        resp = model.generate_content(prompt)
        lines = resp.text.split("\n")
    except Exception as e:
        print(f"Error calling Gemini API for skills: {e}")
        return pd.DataFrame()

    results = []
    header_seen = False
    for raw in lines:
        line = raw.strip()
        if not line or "|-" in line:
            continue
        if not header_seen and ("ชื่องาน" in line and "งานย่อย" in line and "task_id" in line and "subtask_id" in line):
            header_seen = True
            continue
        if '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 5:
            continue
        task_name, sub_task, skill_raw, task_id, subtask_id = parts[:5]
        task_name = normalize_text(task_name).lower()
        sub_task = normalize_text(sub_task).lower()
        normalized = normalize_skill_format(skill_raw)
        if normalized and normalized != "ไม่สามารถจับคู่ได้":
            chosen = []
            for token in [t.strip() for t in normalized.split(',') if t.strip()]:
                best = best_string_match(token, list(skills_pool))
                if best and best not in chosen:
                    chosen.append(best)
            normalized = ", ".join(chosen) if chosen else "ไม่สามารถจับคู่ได้"
        results.append({
            "task_name": task_name,
            "sub_task": sub_task,
            "matched_skill": normalized,
            "project_id": project_df[project_df['task_name'].map(normalize_text).str.lower() == task_name]['project_id'].iloc[0] if not project_df[project_df['task_name'].map(normalize_text).str.lower() == task_name].empty else None,
            "task_id": task_id if task_id else None,
            "subtask_id": subtask_id if subtask_id else None,
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("⚠️ match_skills_to_tasks returned empty DataFrame")
        return df

    pj = project_df.copy()
    pj["task_name"] = pj["task_name"].map(lambda s: normalize_text(s).lower())
    pj["sub_task_name"] = pj["sub_task_name"].map(lambda s: normalize_text(s).lower())

    df = df.merge(
        pj[["task_name", "sub_task_name", "qty", "unit", "start_date", "end_date", "durations_subtask", "project_id", "task_id", "subtask_id"]],
        how="left",
        left_on=["task_name", "sub_task", "project_id"],
        right_on=["task_name", "sub_task_name", "project_id"],
        suffixes=("", "_pj")
    )

    # Preserve task_id and subtask_id from project_df if merge overwrites with None
    df["task_id"] = df.apply(
        lambda row: row["task_id_pj"] if pd.isna(row["task_id"]) and not pd.isna(row["task_id_pj"]) else row["task_id"],
        axis=1
    )
    df["subtask_id"] = df.apply(
        lambda row: row["subtask_id_pj"] if pd.isna(row["subtask_id"]) and not pd.isna(row["subtask_id_pj"]) else row["subtask_id"],
        axis=1
    )

    df = df.drop(columns=["sub_task_name", "task_id_pj", "subtask_id_pj"], errors="ignore")
    print(f"match_skills_to_tasks output shape: {df.shape}, task_id nulls: {df['task_id'].isna().sum()}, subtask_id nulls: {df['subtask_id'].isna().sum()}")
    return df

def match_standards_to_tasks(project_df: pd.DataFrame, standard_df: pd.DataFrame, model: genai.GenerativeModel) -> pd.DataFrame:
    std_pool = standard_df["task_name"].dropna().astype(str).map(normalize_text).str.lower().unique().tolist()
    if not std_pool:
        print("⚠️ No standards found in standard_df.")
        return pd.DataFrame()

    task_lines = [
        f"- งาน: {normalize_text(row['task_name'])}, รายละเอียดงานย่อย: {normalize_text(row['sub_task'])}"
        for _, row in project_df.iterrows()
    ]

    prompt = f"""
คุณเป็นผู้ช่วยจับคู่งานกับมาตรฐานการทำงานจากรายการต่อไปนี้:
{', '.join(std_pool)}

ข้อกำหนดรูปแบบคำตอบ:
- ตาราง Markdown 3 คอลัมน์ (ชื่องาน | งานย่อย | มาตรฐานที่จับคู่ได้)
- ถ้าจับคู่ไม่ได้ ให้ใส่ "ไม่สามารถจับคู่ได้"
"""
    try:
        resp = model.generate_content(prompt + "\nรายการงาน:\n" + "\n".join(task_lines))
        lines = resp.text.split("\n")
    except Exception as e:
        print(f"Error calling Gemini API for standards: {e}")
        return pd.DataFrame()

    rows = []
    header_seen = False
    for raw in lines:
        line = raw.strip()
        if not line or "|-" in line:
            continue
        if not header_seen and ("ชื่องาน" in line and "งานย่อย" in line):
            header_seen = True
            continue
        if '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 3:
            continue
        rows.append({
            "task_name": normalize_text(parts[0]).lower(),
            "sub_task": normalize_text(parts[1]).lower(),
            "matched_standard": normalize_text(parts[2]).lower(),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ match_standards_to_tasks returned empty DataFrame")
        return df

    std = standard_df.copy()
    std["task_name"] = std["task_name"].map(lambda s: normalize_text(s).lower())

    df = df.merge(std, how="left", left_on="matched_standard", right_on="task_name", suffixes=("", "_std"))
    df = df.rename(columns={"standard_rate": "standard_rate", "unit": "standard_unit"})
    df.drop(columns=["task_name_std"], inplace=True, errors="ignore")
    print(f"match_standards_to_tasks output shape: {df.shape}")
    return df

# -------------------- Assignment --------------------
def assign_workers(df_tasks: pd.DataFrame, worker_df: pd.DataFrame) -> pd.DataFrame:
    w = worker_df.copy()
    w["skill_name"] = w["skill_name"].map(lambda s: normalize_text(s).lower())
    w = clean_skill_level(w)
    w["evaluation"] = pd.to_numeric(w["evaluation"], errors="coerce")
    w = w.dropna(subset=["evaluation", "skill_name"])  # keep valid

    skill_to_workers: dict[str, list[dict]] = defaultdict(list)
    for _, r in w.iterrows():
        skill_to_workers[r["skill_name"]].append({
            "worker_id": r["worker_id"],
            "worker_name": r["worker_name"],
            "evaluation": r["evaluation"],
            "skill_level": r["skill_level"],
        })

    out_rows = []
    used_workers = set()

    for _, r in df_tasks.iterrows():
        needed = int(r.get("workers_needed", 1)) if not pd.isna(r.get("workers_needed")) else 1
        matched = str(r.get("matched_skill", "")).strip().lower()
        tokens = [t.strip() for t in matched.split(',') if t.strip()] if matched and matched != "ไม่สามารถจับคู่ได้" else []

        candidate_pool = []
        for tok in tokens:
            best = best_string_match(tok, list(skill_to_workers.keys()))
            if best:
                candidate_pool.extend(skill_to_workers[best])
        candidate_pool = sorted(candidate_pool, key=lambda x: (-float(x["evaluation"]) if not pd.isna(x["evaluation"]) else -0.0))

        assigned_count = 0
        if candidate_pool:
            for cand in candidate_pool:
                if cand["worker_name"] in used_workers:
                    continue
                # Calculate planned_hours as durations_subtask * 8 hours
                planned_hours = r.get("durations_subtask", 0) * 8 if pd.notna(r.get("durations_subtask")) and r.get("durations_subtask", 0) > 0 else 8
                out_rows.append({
                    "assignment_id": str(uuid.uuid4()),  # Generate unique assignment_id
                    "task_name": r.get("task_name"),
                    "sub_task": r.get("sub_task"),
                    "matched_skill": matched if matched else "ไม่สามารถจับคู่ได้",
                    "start_date": r.get("start_date"),
                    "end_date": r.get("end_date"),
                    "unit": r.get("unit"),
                    "qty": r.get("qty"),
                    "durations_subtask": r.get("durations_subtask"),
                    "standard_rate": r.get("standard_rate"),
                    "workers_needed": needed,
                    "worker_id": cand["worker_id"],
                    "worker_name": cand["worker_name"],
                    "evaluation": cand["evaluation"],
                    "project_id": r.get("project_id"),
                    "task_id": r.get("task_id"),
                    "subtask_id": r.get("subtask_id"),
                    "planned_hours": planned_hours,
                    "assigned_by": "system"
                })
                used_workers.add(cand["worker_name"])
                assigned_count += 1
                if assigned_count >= needed:
                    break
        if assigned_count < needed:
            # Calculate planned_hours as durations_subtask * 8 hours for unassigned tasks
            planned_hours = r.get("durations_subtask", 0) * 8 if pd.notna(r.get("durations_subtask")) and r.get("durations_subtask", 0) > 0 else 8
            out_rows.append({
                "assignment_id": str(uuid.uuid4()),  # Generate unique assignment_id
                "task_name": r.get("task_name"),
                "sub_task": r.get("sub_task"),
                "matched_skill": matched if matched else "ไม่สามารถจับคู่ได้",
                "start_date": r.get("start_date"),
                "end_date": r.get("end_date"),
                "unit": r.get("unit"),
                "qty": r.get("qty"),
                "durations_subtask": r.get("durations_subtask"),
                "standard_rate": r.get("standard_rate"),
                "workers_needed": needed,
                "worker_id": None,  # Use None for unassigned workers
                "worker_name": "ไม่สามารถจับคู่ได้",
                "evaluation": np.nan,
                "project_id": r.get("project_id"),
                "task_id": r.get("task_id"),
                "subtask_id": r.get("subtask_id"),
                "planned_hours": planned_hours,
                "assigned_by": "system"
            })

    df_assigned = pd.DataFrame(out_rows)
    print(f"assign_workers output shape: {df_assigned.shape}, task_id nulls: {df_assigned['task_id'].isna().sum()}, subtask_id nulls: {df_assigned['subtask_id'].isna().sum()}")
    return df_assigned

# -------------------- Pipeline --------------------
if __name__ == "__main__":
    print("📥 Loading data from database …")
    worker_df = fetch_query(
        conn,
        worker_query,
        columns=["worker_id", "worker_name", "skill_name", "skill_level", "evaluation"],
    )
    project_df = fetch_query(
        conn,
        project_query,
        columns=["project_id", "project_name", "task_id", "task_name", "subtask_id", "sub_task_name", "qty", "unit", "start_date", "end_date", "durations_subtask"],
    )
    standard_df = fetch_query(
        conn, work_standard_query, columns=["task_name", "standard_rate", "unit"]
    )

    print(f"Loaded project_df with shape: {project_df.shape}, task_id nulls: {project_df['task_id'].isna().sum()}, subtask_id nulls: {project_df['subtask_id'].isna().sum()}")

    project_norm = project_df.copy()
    project_norm["task_name"] = project_norm["task_name"].map(lambda s: normalize_text(s).lower())
    project_norm["sub_task_name"] = project_norm["sub_task_name"].map(lambda s: normalize_text(s).lower())

    print("🔍 Matching skills to tasks with Gemini …")
    df_skills = match_skills_to_tasks(worker_df, project_df, model)
    if df_skills.empty:
        print("⚠️ No skills matched, exiting.")
        sys.exit(1)

    print("🔧 Matching work standards to tasks with Gemini …")
    df_std = match_standards_to_tasks(
        df_skills[["task_name", "sub_task"]].drop_duplicates(),
        standard_df,
        model
    )

    df_merged = df_skills.merge(df_std, on=["task_name", "sub_task"], how="left")
    print(f"df_merged shape: {df_merged.shape}, task_id nulls: {df_merged['task_id'].isna().sum()}, subtask_id nulls: {df_merged['subtask_id'].isna().sum()}")

    # Calculate workers_needed (from provided code)
    df_merged["qty"] = pd.to_numeric(df_merged["qty"], errors="coerce")
    df_merged["standard_rate"] = pd.to_numeric(df_merged["standard_rate"], errors="coerce")
    df_merged["durations_subtask"] = pd.to_numeric(df_merged["durations_subtask"], errors="coerce")
    df_merged["workers_needed"] = df_merged.apply(
        lambda r: int(np.ceil(
            r["qty"] / (r["standard_rate"] * r["durations_subtask"])
        ))
        if pd.notna(r.get("qty"))
           and pd.notna(r.get("standard_rate"))
           and r["standard_rate"] > 0
           and pd.notna(r.get("durations_subtask"))
           and r["durations_subtask"] > 0
        else 1,
        axis=1,
    )

    print("👷 Assigning workers …")
    df_assigned = assign_workers(df_merged, worker_df)

    out1 = os.path.join(os.getcwd(), "matched_skills.csv")
    out2 = os.path.join(os.getcwd(), "matched_standards.csv")
    out3 = os.path.join(os.getcwd(), "assigned_workers.csv")

    df_skills.to_csv(out1, index=False, encoding="utf-8")
    df_std.to_csv(out2, index=False, encoding="utf-8")
    df_assigned.to_csv(out3, index=False, encoding="utf-8")

    print("✅ Done. Saved:")
    print(" -", out1)
    print(" -", out2)
    print(" -", out3)

    worker_ids = df_assigned[df_assigned["worker_id"].notna()]["worker_id"].unique().tolist()

    with engine.begin() as conn:
        if worker_ids:
            conn.execute(
                text("""
                    UPDATE worker_status
                    SET status = 'ไม่ว่าง'
                    WHERE worker_id = ANY(:ids)
                """),
                {"ids": worker_ids}
            )
            print(f"Updated worker_status for {len(worker_ids)} workers to 'ไม่ว่าง'")

    # Prepare data for Worker_Assignment table
    df_to_upload = df_assigned[["assignment_id", "worker_id", "project_id", "task_id", "subtask_id", "start_date", "end_date", "planned_hours", "assigned_by"]].copy()

    # Ensure data types match the Worker_Assignment table schema
    df_to_upload["assignment_id"] = df_to_upload["assignment_id"].astype(str)
    df_to_upload["worker_id"] = df_to_upload["worker_id"].astype(str, errors="ignore").replace("nan", None)
    df_to_upload["project_id"] = df_to_upload["project_id"].astype(str, errors="ignore").replace("nan", None)
    df_to_upload["task_id"] = df_to_upload["task_id"].astype(str, errors="ignore").replace("nan", None)
    df_to_upload["subtask_id"] = df_to_upload["subtask_id"].astype(str, errors="ignore").replace("nan", None)
    df_to_upload["start_date"] = pd.to_datetime(df_to_upload["start_date"], errors="coerce")
    df_to_upload["end_date"] = pd.to_datetime(df_to_upload["end_date"], errors="coerce")
    df_to_upload["planned_hours"] = pd.to_numeric(df_to_upload["planned_hours"], errors="coerce")
    df_to_upload["assigned_by"] = df_to_upload["assigned_by"].astype(str)

    # Log rows with null task_id or subtask_id
    null_task_ids = df_to_upload[df_to_upload["task_id"].isna()]
    null_subtask_ids = df_to_upload[df_to_upload["subtask_id"].isna()]
    if not null_task_ids.empty:
        print(f"⚠️ {len(null_task_ids)} rows with null task_id: {null_task_ids[['task_name', 'sub_task']].to_dict('records')}")
    if not null_subtask_ids.empty:
        print(f"⚠️ {len(null_subtask_ids)} rows with null subtask_id: {null_subtask_ids[['task_name', 'sub_task']].to_dict('records')}")
    
    # Drop rows with null task_id, subtask_id, or project_id to satisfy foreign key constraints
    df_to_upload = df_to_upload.dropna(subset=["project_id", "task_id", "subtask_id"])
    print(f"df_to_upload shape after dropping nulls: {df_to_upload.shape}")

    # Verify foreign key constraints
    with engine.begin() as conn:
        # Check if task_id and subtask_id exist in Tasks and Subtask tables
        valid_task_ids = conn.execute(text("SELECT task_id FROM Tasks")).fetchall()
        valid_task_ids = {row[0] for row in valid_task_ids}
        valid_subtask_ids = conn.execute(text("SELECT subtask_id FROM Subtask")).fetchall()
        valid_subtask_ids = {row[0] for row in valid_subtask_ids}

        invalid_tasks = df_to_upload[~df_to_upload["task_id"].isin(valid_task_ids)]
        invalid_subtasks = df_to_upload[~df_to_upload["subtask_id"].isin(valid_subtask_ids)]
        if not invalid_tasks.empty:
            print(f"⚠️ Invalid task_ids: {invalid_tasks['task_id'].unique().tolist()}")
        if not invalid_subtasks.empty:
            print(f"⚠️ Invalid subtask_ids: {invalid_subtasks['subtask_id'].unique().tolist()}")
        
        # Filter out invalid IDs
        df_to_upload = df_to_upload[df_to_upload["task_id"].isin(valid_task_ids) & df_to_upload["subtask_id"].isin(valid_subtask_ids)]

    # อัปโหลดเข้า Neon DB
    try:
        df_to_upload.to_sql(
            name="Worker_Assignment",
            con=engine,
            if_exists="append",
            index=False
        )
        print(f"✅ Successfully uploaded {len(df_to_upload)} rows to Worker_Assignment table")
    except Exception as e:
        print(f"❌ Error uploading to Worker_Assignment: {e}")
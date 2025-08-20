import re
import pandas as pd
import google.generativeai as genai
from .utils import normalize_text, normalize_skill_format, best_string_match

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

def parse_skill_level(val):
    text = str(val).strip() if not pd.isna(val) else ""
    if re.search(r"ทดลอง|ฝึกงาน|intern", text, re.IGNORECASE):
        return 0.0, "U"
    m = re.compile(r"(\d+(?:\.\d+)?)").search(text)
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
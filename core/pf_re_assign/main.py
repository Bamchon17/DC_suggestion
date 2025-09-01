import pandas as pd
import math
import sys
import os
import re
import difflib
import numpy as np
import psycopg2
from collections import defaultdict
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import uuid

# -------------------- Path & ENV --------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)
from db.connection import conn  # expects a `conn` psycopg2 connection

load_dotenv('db.env')

# ดึงค่าจาก .env
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# สร้าง Connection URL สำหรับ Postgres (Neon ใช้ sslmode=require)
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"

# สร้าง engine
engine = create_engine(DATABASE_URL)

# -------------------- Utilities --------------------
TH_SEP_PATTERN = re.compile(r'(;และ|，|、|：|\s+และ\s+)')
NON_ALNUM_TH = re.compile(r'[^0-9a-zA-Zก-๙\s,]')
MULTI_SPACE = re.compile(r'\s+')

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

# -------------------- Skill level parsing --------------------
INT_OR_FLOAT = re.compile(r'(\d+(\.\d+)?)')

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

# -------------------- อ่านไฟล์ CSV และคำนวณคนงานใหม่ --------------------
pf_alerts = pd.read_csv('pf_alerts.csv')
matched_skills = pd.read_csv('matched_skills.csv')

# ลบแถวที่ซ้ำใน pf_alerts โดยใช้ subtask_id
pf_alerts_unique = pf_alerts.drop_duplicates(subset=['subtask_id'], keep='first')

# รวมข้อมูลจาก pf_alerts และ matched_skills โดยใช้ subtask_id
merged_df = pf_alerts_unique.merge(
    matched_skills[['subtask_id', 'matched_skill', 'task_name', 'qty', 'unit', 'start_date', 'end_date', 'durations_subtask']],
    on='subtask_id',
    how='left'
)

# คำนวณคนงานใหม่
results = []
for index, row in merged_df.iterrows():
    subtask_id = row['subtask_id']
    sub_task_name = row['sub_task_name']
    planned_hours = row['planned_hours']
    hours_worked = row['hours_worked']
    qty = row['qty_x']  # จาก pf_alerts
    unit_completed = row['unit_completed']
    num_workers = row['num_workers']
    matched_skill = row['matched_skill'] if pd.notna(row['matched_skill']) else 'ไม่ระบุทักษะ'
    task_name = row['task_name'] if pd.notna(row['task_name']) else 'ไม่ระบุงานหลัก'
    start_date = row['start_date']
    end_date = row['end_date']
    durations_subtask = row['durations_subtask']
    unit = row['unit']
    
    # คำนวณชั่วโมงที่เหลือและหน่วยงานที่เหลือ
    remaining_hours = planned_hours - hours_worked
    remaining_units = qty - unit_completed
    
    # ข้ามกรณีที่ hours_worked หรือ unit_completed เป็น 0
    if hours_worked == 0 or unit_completed == 0:
        new_workers = 0
    else:
        # คำนวณประสิทธิภาพต่อคนต่อชั่วโมง
        efficiency_per_worker = (unit_completed / hours_worked) / num_workers
        
        # ถ้า remaining_hours หรือ remaining_units <= 0 ไม่ต้องเพิ่มคนงาน
        if remaining_hours <= 0 or remaining_units <= 0:
            new_workers = 0
        else:
            # คำนวณจำนวนคนงานที่ต้องการทั้งหมด
            required_workers = (remaining_units / remaining_hours) / efficiency_per_worker
            # ปัดขึ้นเพื่อให้ได้จำนวนคนงานเต็ม
            required_workers = math.ceil(required_workers)
            # คำนวณจำนวนคนงานใหม่
            new_workers = max(0, required_workers - num_workers)
    
    # เก็บผลลัพธ์
    results.append({
        'subtask_id': subtask_id,
        'task_name': task_name,
        'sub_task': sub_task_name,
        'num_workers': num_workers,
        'remaining_hours': round(remaining_hours, 2),
        'remaining_units': round(remaining_units, 2),
        'workers_needed': new_workers,
        'matched_skill': matched_skill,
        'start_date': start_date,
        'end_date': end_date,
        'durations_subtask': durations_subtask,
        'qty': qty,
        'unit': unit
    })

# สร้าง DataFrame จากผลลัพธ์
results_df = pd.DataFrame(results)

# กรองเฉพาะงานที่ต้องเพิ่มคนงาน (workers_needed > 0)
df_tasks = results_df[results_df['workers_needed'] > 0]

# -------------------- ดึงข้อมูลคนงานจากฐานข้อมูล --------------------
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

worker_df = fetch_query(
    conn,
    worker_query,
    columns=['worker_id', 'worker_name', 'skill_name', 'skill_level', 'evaluation']
)

# -------------------- มอบหมายคนงาน --------------------
def assign_workers(df_tasks: pd.DataFrame, worker_df: pd.DataFrame) -> pd.DataFrame:
    w = worker_df.copy()
    w['skill_name'] = w['skill_name'].map(lambda s: normalize_text(s).lower())
    w = clean_skill_level(w)
    w['evaluation'] = pd.to_numeric(w['evaluation'], errors='coerce')
    w = w.dropna(subset=['evaluation', 'skill_name'])  # keep valid

    skill_to_workers: dict[str, list[dict]] = defaultdict(list)
    for _, r in w.iterrows():
        skill_to_workers[r['skill_name']].append({
            'worker_id': r['worker_id'],
            'worker_name': r['worker_name'],
            'evaluation': r['evaluation'],
            'skill_level': r['skill_level'],
        })

    out_rows = []
    used_workers = set()

    for _, r in df_tasks.iterrows():
        needed = int(r.get('workers_needed', 1))
        matched = str(r.get('matched_skill', '')).strip().lower()
        tokens = [t.strip() for t in matched.split(',') if t.strip()] if matched and matched != 'ไม่ระบุทักษะ' else []

        candidate_pool = []
        for tok in tokens:
            if tok in skill_to_workers:
                candidate_pool.extend(skill_to_workers[tok])
        candidate_pool = sorted(candidate_pool, key=lambda x: (-float(x['evaluation']) if not pd.isna(x['evaluation']) else -0.0))

        if tokens and not candidate_pool:
            print(f"⚠️ Matched skill '{matched}' but no candidates found for task {r.get('task_name')}, sub_task {r.get('sub_task')}")

        assigned_count = 0
        if candidate_pool:
            for cand in candidate_pool:
                if cand['worker_name'] in used_workers:
                    continue
                planned_hours = r.get('durations_subtask', 0) * 8 if pd.notna(r.get('durations_subtask')) and r.get('durations_subtask', 0) > 0 else 8
                out_rows.append({
                    'assignment_id': str(uuid.uuid4()),
                    'task_name': r.get('task_name'),
                    'sub_task': r.get('sub_task'),
                    'matched_skill': matched if matched else 'ไม่ระบุทักษะ',
                    'start_date': r.get('start_date'),
                    'end_date': r.get('end_date'),
                    'unit': r.get('unit'),
                    'qty': r.get('qty'),
                    'durations_subtask': r.get('durations_subtask'),
                    'workers_needed': needed,
                    'worker_id': cand['worker_id'],
                    'worker_name': cand['worker_name'],
                    'evaluation': cand['evaluation'],
                    'project_id': '12',  # จาก matched_skills.csv
                    'task_id': r.get('subtask_id').replace('SUB', 'AIR'),  # สมมติ task_id จาก subtask_id
                    'subtask_id': r.get('subtask_id'),
                    'planned_hours': planned_hours,
                    'assigned_by': 'system'
                })
                used_workers.add(cand['worker_name'])
                assigned_count += 1
                if assigned_count == needed:
                    break
        if assigned_count < needed:
            planned_hours = r.get('durations_subtask', 0) * 8 if pd.notna(r.get('durations_subtask')) and r.get('durations_subtask', 0) > 0 else 8
            out_rows.append({
                'assignment_id': str(uuid.uuid4()),
                'task_name': r.get('task_name'),
                'sub_task': r.get('sub_task'),
                'matched_skill': matched if matched else 'ไม่ระบุทักษะ',
                'start_date': r.get('start_date'),
                'end_date': r.get('end_date'),
                'unit': r.get('unit'),
                'qty': r.get('qty'),
                'durations_subtask': r.get('durations_subtask'),
                'workers_needed': needed,
                'worker_id': None,
                'worker_name': 'ไม่สามารถจับคู่ได้',
                'evaluation': np.nan,
                'project_id': '12',
                'task_id': r.get('subtask_id').replace('SUB', 'AIR'),
                'subtask_id': r.get('subtask_id'),
                'planned_hours': planned_hours,
                'assigned_by': 'system'
            })

    df_assigned = pd.DataFrame(out_rows)
    print(f"assign_workers output shape: {df_assigned.shape}, task_id nulls: {df_assigned['task_id'].isna().sum()}, subtask_id nulls: {df_assigned['subtask_id'].isna().sum()}")
    return df_assigned

# มอบหมายคนงาน
df_assigned = assign_workers(df_tasks, worker_df)

# -------------------- บันทึกและอัปเดตฐานข้อมูล --------------------
# บันทึกผลลัพธ์เป็น CSV
output_file = os.path.join(os.getcwd(), 'assigned_new_workers.csv')
df_assigned.to_csv(output_file, index=False, encoding='utf-8')
print(f"✅ บันทึกผลลัพธ์ที่: {output_file}")
"""
# เตรียมข้อมูลสำหรับอัปเดต Worker_Assignment
df_to_upload = df_assigned[[
    'assignment_id', 'worker_id', 'project_id', 'task_id', 
    'subtask_id', 'start_date', 'end_date', 'planned_hours', 'assigned_by'
]].copy()

# แปลงประเภทข้อมูลให้ตรงกับตาราง Worker_Assignment
df_to_upload['assignment_id'] = df_to_upload['assignment_id'].astype(str)
df_to_upload['worker_id'] = df_to_upload['worker_id'].astype(str, errors='ignore').replace('nan', None)
df_to_upload['project_id'] = df_to_upload['project_id'].astype(str, errors='ignore').replace('nan', None)
df_to_upload['task_id'] = df_to_upload['task_id'].astype(str, errors='ignore').replace('nan', None)
df_to_upload['subtask_id'] = df_to_upload['subtask_id'].astype(str, errors='ignore').replace('nan', None)
df_to_upload['start_date'] = pd.to_datetime(df_to_upload['start_date'], errors='coerce')
df_to_upload['end_date'] = pd.to_datetime(df_to_upload['end_date'], errors='coerce')
df_to_upload['planned_hours'] = pd.to_numeric(df_to_upload['planned_hours'], errors='coerce')
df_to_upload['assigned_by'] = df_to_upload['assigned_by'].astype(str)

# ตรวจสอบ foreign key constraints
with engine.begin() as conn:
    valid_task_ids = conn.execute(text("SELECT task_id FROM Tasks")).fetchall()
    valid_task_ids = {row[0] for row in valid_task_ids}
    valid_subtask_ids = conn.execute(text("SELECT subtask_id FROM Subtask")).fetchall()
    valid_subtask_ids = {row[0] for row in valid_subtask_ids}

    invalid_tasks = df_to_upload[~df_to_upload['task_id'].isin(valid_task_ids)]
    invalid_subtasks = df_to_upload[~df_to_upload['subtask_id'].isin(valid_subtask_ids)]
    if not invalid_tasks.empty:
        print(f"⚠️ Invalid task_ids: {invalid_tasks['task_id'].unique().tolist()}")
    if not invalid_subtasks.empty:
        print(f"⚠️ Invalid subtask_ids: {invalid_subtasks['subtask_id'].unique().tolist()}")
    
    # กรองเฉพาะข้อมูลที่ถูกต้อง
    df_to_upload = df_to_upload[df_to_upload['task_id'].isin(valid_task_ids) & df_to_upload['subtask_id'].isin(valid_subtask_ids)]

    # อัปเดตฐานข้อมูล
    if not df_to_upload.empty:
        df_to_upload.to_sql('Worker_Assignment', conn, if_exists='append', index=False)
        print(f"✅ อัปเดตตาราง Worker_Assignment ด้วย {len(df_to_upload)} แถว")

    # อัปเดตสถานะคนงาน
    worker_ids = df_to_upload[df_to_upload['worker_id'].notna()]['worker_id'].unique().tolist()
    if worker_ids:
        conn.execute(
            text("UPDATE worker_status SET status = 'ไม่ว่าง' WHERE worker_id = ANY(:ids)"),
            {'ids': worker_ids}
        )
        print(f"✅ อัปเดตสถานะ worker_status สำหรับ {len(worker_ids)} คนงานเป็น 'ไม่ว่าง'")

print("📤 เสร็จสิ้นการประมวลผลและอัปเดตฐานข้อมูล")
"""
# core/manday_calculation/pipeline.py
import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text

# ---------------- Gemini API ----------------
import google.generativeai as genai
from dotenv import load_dotenv

# โหลด .env
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# ตั้งค่า Gemini API key จาก .env
api_key = os.getenv("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("No Gemini API key found in environment variables!")
genai.configure(api_key=api_key)
print("✅ Gemini API configured successfully.")

# ---------------- Existing imports ----------------
from core.manday_calculation.utils import normalize_text, model
from core.manday_calculation.skill_matching import match_skills_to_tasks, worker_query
from core.manday_calculation.standard_matching import match_standards_to_tasks, work_standard_query
from core.manday_calculation.assignment import assign_workers
from core.manday_calculation.queries import worker_query, project_query, work_standard_query

from db.connection import get_connection, get_engine, fetch_query

# database connection
conn = get_connection()
engine = get_engine()

# -------------------- Pipeline --------------------

if __name__ == '__main__':
    print("📥 Loading data from database …")
    worker_df = fetch_query(worker_query)[
        ['worker_id', 'worker_name', 'skill_name', 'skill_level', 'evaluation']
    ]

    project_df = fetch_query(project_query)[
        ['project_id', 'project_name', 'task_id', 'task_name', 'subtask_id', 'sub_task_name', 'qty', 'unit', 'start_date', 'end_date', 'durations_subtask']
    ]

    standard_df = fetch_query(work_standard_query)[
        ['task_name', 'standard_rate', 'unit']
    ]
    
    print(f"Loaded project_df with shape {project_df.shape}, task_id nulls {project_df['task_id'].isna().sum()}, subtask_id nulls {project_df['subtask_id'].isna().sum()}")

    project_norm = project_df.copy()
    project_norm['task_name'] = project_norm['task_name'].map(lambda s: normalize_text(s).lower())
    project_norm['sub_task_name'] = project_norm['sub_task_name'].map(lambda s: normalize_text(s).lower())

    print("🔍 Matching skills to tasks with Gemini …")
    df_skills = match_skills_to_tasks(worker_df, project_df, model)
    if df_skills.empty:
        print("⚠️ No skills matched, exiting.")
        sys.exit(1)

    print("🔧 Matching work standards to tasks with Gemini …")
    df_std = match_standards_to_tasks(
        df_skills[['task_name', 'sub_task']].drop_duplicates(),
        standard_df,
        model
    )

    df_merged = df_skills.merge(df_std, on=['task_name', 'sub_task'], how='left')
    print(f"df_merged shape: {df_merged.shape}, task_id nulls: {df_merged['task_id'].isna().sum()}, subtask_id nulls: {df_merged['subtask_id'].isna().sum()}")

    df_merged['qty'] = pd.to_numeric(df_merged['qty'], errors='coerce')
    df_merged['standard_rate'] = pd.to_numeric(df_merged['standard_rate'], errors='coerce')
    df_merged['durations_subtask'] = pd.to_numeric(df_merged['durations_subtask'], errors='coerce')
    df_merged['workers_needed'] = df_merged.apply(
        lambda r: int(np.ceil(
            r['qty'] / (r['standard_rate'] * r['durations_subtask'])
        ))
        if pd.notna(r.get('qty')) 
           and pd.notna(r.get('standard_rate'))
           and r['standard_rate'] > 0
           and pd.notna(r.get('durations_subtask'))
           and r['durations_subtask'] > 0
        else 1,
        axis=1,
    )  

    print("👷 Assigning workers …")
    df_assigned = assign_workers(df_merged, worker_df)

    out1 = os.path.join(os.getcwd(), 'matched_skills.csv')
    out2 = os.path.join(os.getcwd(), 'matched_standards.csv')
    out3 = os.path.join(os.getcwd(), 'assigned_workers.csv')

    df_skills.to_csv(out1, index=False, encoding='utf-8')
    df_std.to_csv(out2, index=False, encoding='utf-8')
    df_assigned.to_csv(out3, index=False, encoding='utf-8')

    print("✅ Done. Saved:")
    print(f" - {out1}")
    print(f" - {out2}")
    print(f" - {out3}")

    # -------------------- Database Update --------------------
    df_to_upload = df_assigned[['assignment_id', 'worker_id', 'project_id', 'task_id', 'subtask_id', 'start_date', 'end_date', 'planned_hours', 'assigned_by', 'skill_level_num']].copy()

    # --- แก้ไขให้เหมือนเพื่อน ---
    df_to_upload['planned_hours'] = pd.to_numeric(df_to_upload['planned_hours'], errors='coerce')
    df_to_upload['skill_level_num'] = pd.to_numeric(df_to_upload['skill_level_num'], errors='coerce')
    df_to_upload['worker_id'] = df_to_upload['worker_id'].replace('None', None)

    # เรียงลำดับ deterministic
    df_to_upload = df_to_upload.sort_values(['project_id', 'task_id', 'subtask_id', 'worker_id']).reset_index(drop=True)

    # ตั้ง assignment_id sequential แบบเพื่อน
    df_to_upload['assignment_id'] = df_to_upload.index + 1
    df_to_upload['assignment_id'] = df_to_upload['assignment_id'].astype(str)

    print(f"df_to_upload shape before foreign key validation: {df_to_upload.shape}")

    # Validate foreign keys
    with engine.begin() as conn:
        valid_task_ids = {row[0] for row in conn.execute(text("SELECT task_id FROM Tasks")).fetchall()}
        valid_subtask_ids = {row[0] for row in conn.execute(text("SELECT subtask_id FROM Subtask")).fetchall()}

        df_to_upload = df_to_upload[df_to_upload['task_id'].isin(valid_task_ids) & df_to_upload['subtask_id'].isin(valid_subtask_ids)]
        print(f"df_to_upload shape after foreign key validation: {df_to_upload.shape}")

        if not df_to_upload.empty:
            df_to_upload.to_sql('worker_assignment', conn, if_exists='append', index=False)
            print(f"✅ Successfully updated Worker_Assignment with {len(df_to_upload)} rows")
        else:
            print("⚠️ No valid rows to upload to Worker_Assignment")

        worker_ids = df_to_upload[df_to_upload['worker_id'].notna()]['worker_id'].unique().tolist()
        if worker_ids:
            conn.execute(
                text("UPDATE worker_status SET status = 'ไม่ว่าง' WHERE worker_id = ANY(:ids)"),
                {'ids': worker_ids}
            )
            print(f"✅ Updated worker_status for {len(worker_ids)} workers to 'ไม่ว่าง'")

    print("📤 Finished processing and uploading assignments.")

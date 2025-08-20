# core/pf_calculation/calculate_pf.py
from db.connection import fetch_query, conn
import pandas as pd
from datetime import datetime
import uuid

ALERT_THRESHOLD = 0.9

def calculate_pf(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # แปลงคอลัมน์เป็น float และ fillna
    for col in ['unit_completed','qty','durations_subtask','workers_assigned','standard_rate']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)
        else:
            df[col] = 0.0

    # 1. Planned Hours
    df['planned_hours'] = df['durations_subtask'] * 8 * df['workers_assigned']

    # 2. Hours Worked (handle division by zero)
    df['hours_worked'] = df.apply(
        lambda x: x['unit_completed'] / x['standard_rate'] if x['standard_rate'] > 0 else 0,
        axis=1
    )

    # 3. PF Time (handle division by zero)
    df['pf_time'] = df.apply(
        lambda x: x['planned_hours'] / x['hours_worked'] if x['hours_worked'] > 0 else 0,
        axis=1
    )

    # 4. PF Quantity
    df['pf_qty'] = df.apply(
        lambda x: x['unit_completed'] / x['qty'] if x['qty'] > 0 else 0,
        axis=1
    )

    # 5. Alert Flag
    df['alert_flag'] = (df['pf_time'] < ALERT_THRESHOLD) | (df['pf_qty'] < ALERT_THRESHOLD)

    # 6. Recommend Worker
    df['recommended_worker_id'] = None
    df['recommended_worker_name'] = None
    alert_df = df[df['alert_flag']]

    for idx, row in alert_df.iterrows():
        rec_query = f"""
        SELECT w.worker_id, w.worker_name
        FROM Worker w
        JOIN Worker_Status ws ON w.worker_id = ws.worker_id
        JOIN Skill_Record sr ON w.worker_id = sr.worker_id
        JOIN Skill_Type st ON sr.skill_type_id = st.skill_type_id
        WHERE ws.status = 'ว่าง'
          AND st.skill_name = '{row['sub_task_name']}'
        LIMIT 1
        """
        rec = fetch_query(conn, rec_query)
        if not rec.empty:
            df.at[idx, 'recommended_worker_id'] = rec.iloc[0]['worker_id']
            df.at[idx, 'recommended_worker_name'] = rec.iloc[0]['worker_name']

    # 7. Log ID, Date, Notes
    df['log_id'] = df.apply(lambda x: str(uuid.uuid4()), axis=1)
    df['date'] = datetime.today().date()
    df['notes'] = None

    # เลือกคอลัมน์ตาม schema ใหม่
    pf_log_df = df[['log_id','date','project_id','task_id','subtask_id','worker_id',
                    'planned_hours','hours_worked','pf_time','pf_qty','alert_flag',
                    'recommended_worker_id','recommended_worker_name','notes']]

    return pf_log_df

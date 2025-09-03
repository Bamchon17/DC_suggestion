# core/pf_calculation/calculate_pf.py
import pandas as pd
import numpy as np
from .fetcher import fetch_assignments, fetch_worklogs, fetch_subtasks
from db.connection import get_engine
import uuid
import datetime

MAX_WORKERS_DISPLAY = 5  # จำนวน worker ที่จะแสดงตรง alert

def format_worker_list(workers: list) -> str:
    if not workers:
        return ""
    if len(workers) <= MAX_WORKERS_DISPLAY:
        return ", ".join(workers)
    else:
        displayed = ", ".join(workers[:MAX_WORKERS_DISPLAY])
        remaining = len(workers) - MAX_WORKERS_DISPLAY
        return f"{displayed} +{remaining} more"

def save_pf_log(df: pd.DataFrame, table_name="pf_log"):
    """บันทึกทุก PF ลง DB"""
    engine = get_engine()

    # map column ให้ตรงกับ DB
    db_df = df.copy()
    db_df = db_df.rename(columns={
        "hours_worked": "actual_hours",
        "qty": "qty_total"
    })

    # สร้าง pf_id uuid
    db_df['pf_id'] = [str(uuid.uuid4()) for _ in range(len(db_df))]

    # drop columns ที่ DB ไม่มี
    drop_cols = ['worker_ids', 'worker_display', 'sub_task_name', 'num_workers']
    db_df = db_df.drop(columns=[c for c in drop_cols if c in db_df.columns])

    # save ลง DB
    db_df.to_sql(table_name, con=engine, if_exists='append', index=False)
    print(f"Saved {len(db_df)} PF records to {table_name}")


def calculate_daily_pf(project_id: str | None = None) -> pd.DataFrame:
    # ดึงข้อมูล
    assignments = fetch_assignments(project_id)
    worklogs = fetch_worklogs(project_id)
    subtasks = fetch_subtasks(project_id)

    if assignments.empty:
        print("No assignments found")
        return pd.DataFrame()

    # รวม worklogs ต่อ task/subtask
    total_work = worklogs.groupby(['project_id', 'task_id', 'subtask_id']).agg(
        hours_worked=('hours_worked', 'sum'),
        unit_completed=('unit_completed', 'sum'),
        worker_ids=('worker_id', lambda x: list(x.unique())),
        num_workers=('worker_id', 'nunique')
    ).reset_index()

    # merge assignments กับ worklogs → left join
    pf_df = assignments.merge(
        total_work,
        on=['project_id', 'task_id', 'subtask_id'],
        how='left'
    )

    # fallback
    pf_df['hours_worked'] = pf_df['hours_worked'].fillna(0)
    pf_df['unit_completed'] = pf_df['unit_completed'].fillna(0)
    pf_df['worker_ids'] = pf_df['worker_ids'].apply(lambda x: x if isinstance(x, list) and x else [])
    pf_df['num_workers'] = pf_df['num_workers'].fillna(0)

    # merge ชื่อ subtask + qty
    pf_df = pf_df.merge(subtasks[['subtask_id', 'sub_task_name', 'qty']], on='subtask_id', how='left')
    pf_df['qty'] = pf_df['qty'].fillna(0)

    # convert เป็น float
    for col in ['planned_hours', 'hours_worked', 'unit_completed', 'qty']:
        pf_df[col] = pf_df[col].astype(float)

    # คำนวณ PF
    pf_df['pf_time'] = pf_df.apply(
        lambda x: round(x['hours_worked'] / x['planned_hours'], 2) if x['planned_hours'] > 0 else None,
        axis=1
    )
    pf_df['pf_qty'] = pf_df.apply(
        lambda x: round(x['unit_completed'] / x['qty'], 4) if x['qty'] > 0 else None,
        axis=1
    )

    # log_date = วันนี้
    pf_df['log_date'] = pd.to_datetime('today').date()

    # format worker list ให้สวย
    pf_df['worker_display'] = pf_df['worker_ids'].apply(format_worker_list)

    # save ทุก PF ลง DB
    save_pf_log(pf_df)

    # filter เฉพาะ alert (PF < 1) สำหรับ CSV
    alert_df = pf_df[(pf_df['pf_time'] < 1) | (pf_df['pf_qty'] < 1)].reset_index(drop=True)

    # save CSV
    alert_df.to_csv(f"pf_alert_project_{project_id}_{datetime.date.today()}.csv", index=False)
    print(f"Saved {len(alert_df)} PF alerts to CSV")

    return alert_df

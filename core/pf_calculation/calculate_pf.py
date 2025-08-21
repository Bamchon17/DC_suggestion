# core/pf_calculation/calculate_pf.py
import pandas as pd
import numpy as np
from .fetcher import fetch_assignments, fetch_worklogs, fetch_subtasks

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

    # merge assignments กับ worklogs → left join เพื่อเก็บงานทุกงาน
    pf_df = assignments.merge(
        total_work,
        on=['project_id', 'task_id', 'subtask_id'],
        how='left'
    )

    # เตรียมค่า fallback
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

    # alert = PF เวลา หรือ จำนวน < 1
    pf_df['alert'] = (
        ((pf_df['pf_time'] < 1) & pf_df['pf_time'].notna()) |
        ((pf_df['pf_qty'] < 1) & pf_df['pf_qty'].notna())
    )

    # log_date = วันนี้
    pf_df['log_date'] = pd.to_datetime('today').date()

    return pf_df

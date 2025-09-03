import pandas as pd
import numpy as np
from datetime import datetime
from db.connection import get_connection
from psycopg2.extras import execute_batch

# ---------------- DB ----------------
conn = get_connection()

# ดึง assignment + qty
query = """
SELECT wa.*, s.qty
FROM worker_assignment wa
LEFT JOIN subtask s ON wa.subtask_id = s.subtask_id
"""
assignments = pd.read_sql(query, conn)
workers = pd.read_sql("SELECT * FROM worker", conn)

# ---------------- Filter valid workers ----------------
assignments = assignments[assignments['worker_id'].notnull()]
assignments = assignments[assignments['worker_id'].isin(workers['worker_id'])]

# ---------------- Check required columns ----------------
required_cols = ['assignment_id', 'worker_id', 'project_id', 'task_id', 'subtask_id', 
                 'start_date', 'end_date', 'planned_hours', 'assigned_by']
missing = [c for c in required_cols if c not in assignments.columns]
if missing:
    raise ValueError(f"Missing columns in worker_assignment: {missing}")

# ---------------- Default qty ----------------
if 'qty' not in assignments.columns:
    assignments['qty'] = 100

# ---------------- Adjust planned_hours per worker ----------------
subtask_counts = assignments.groupby('subtask_id')['worker_id'].transform('count')
assignments['planned_hours_per_worker'] = assignments['planned_hours'] / subtask_counts

# ---------------- Worklog Generator ----------------
def generate_worklog(row, log_counter, productivity_rate=5, work_days_ratio=0.8):
    worklogs = []

    start_date = pd.to_datetime(row['start_date']) if pd.notnull(row['start_date']) else pd.Timestamp.today()
    end_date = pd.to_datetime(row['end_date']) if pd.notnull(row['end_date']) else start_date

    planned_hours = float(row['planned_hours_per_worker'])
    qty_total = float(row['qty']) if pd.notnull(row['qty']) else 100

    total_days = max((end_date - start_date).days + 1, 1)
    work_days = min(int(total_days * work_days_ratio), total_days)
    if work_days <= 0:
        return worklogs, log_counter

    all_dates = pd.date_range(start_date, end_date, freq='B')
    if len(all_dates) > work_days:
        indices = np.linspace(0, len(all_dates) - 1, work_days, dtype=int)
        selected_dates = all_dates[indices]
    else:
        selected_dates = all_dates

    remaining_hours = planned_hours
    remaining_units = qty_total

    for log_date in selected_dates:
        hours = round(
            min(
                np.random.uniform(max(planned_hours/work_days*0.8, 4), min(planned_hours/work_days*1.2, 8)),
                remaining_hours
            ),
            2
        )
        if hours <= 0:
            break

        units = round(
            min(
                hours * productivity_rate * np.random.uniform(0.8, 1.2),
                remaining_units
            ),
            2
        )
        if units <= 0:
            break

        log_id = f"LOG{log_counter:05d}"
        log_counter += 1

        worklogs.append({
            'log_id': log_id,
            'worker_id': row['worker_id'],
            'project_id': row['project_id'],
            'task_id': row['task_id'],
            'subtask_id': row['subtask_id'],
            'unit_completed': units,
            'log_date': pd.Timestamp(log_date),
            'hours_worked': hours
        })

        remaining_hours -= hours
        remaining_units -= units

        if remaining_hours <= 0 or remaining_units <= 0:
            break

    return worklogs, log_counter

# ---------------- MAIN ----------------
worklog_list = []
log_counter = 1

for _, row in assignments.iterrows():
    worklogs, log_counter = generate_worklog(row, log_counter, productivity_rate=5, work_days_ratio=0.8)
    worklog_list.extend(worklogs)

worklog_df = pd.DataFrame(worklog_list)

# ---------------- Insert batch ----------------
if not worklog_df.empty:
    cur = conn.cursor()
    insert_query = """
    INSERT INTO worklog (log_id, worker_id, project_id, task_id, subtask_id, unit_completed, log_date, hours_worked)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    data = worklog_df[['log_id', 'worker_id', 'project_id', 'task_id', 'subtask_id', 'unit_completed', 'log_date', 'hours_worked']].values.tolist()
    execute_batch(cur, insert_query, data, page_size=100)
    conn.commit()
    cur.close()

conn.close()

print(f"Inserted {len(worklog_df)} worklog records.")

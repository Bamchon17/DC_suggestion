import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from db.connection import get_connection

# ดึง worker_assignment ทั้งหมด
conn = get_connection()
assignments = pd.read_sql("SELECT * FROM worker_assignment", conn)
workers = pd.read_sql("SELECT * FROM worker", conn)

# check missing worker_id
assignments = assignments[assignments['worker_id'].isin(workers['worker_id'])]


# ตรวจสอบ columns ที่จำเป็น
required_cols = ['assignment_id', 'worker_id', 'project_id', 'task_id', 'subtask_id', 
                 'start_date', 'end_date', 'planned_hours', 'assigned_by']
missing = [c for c in required_cols if c not in assignments.columns]
if missing:
    raise ValueError(f"Missing columns in worker_assignment: {missing}")

# ฟังก์ชันสร้าง worklog สำหรับแต่ละ assignment
def generate_worklog(row, log_counter):
    worklogs = []
    start_date = pd.to_datetime(row['start_date'])
    end_date = pd.to_datetime(row['end_date'])
    planned_hours = float(row['planned_hours'])
    
    # จำนวนวันทั้งหมด
    total_days = max((end_date - start_date).days + 1, 1)
    work_days = min(np.random.randint(3, 6), total_days)  # 3-5 วัน หรือเต็มช่วง
    
    all_dates = pd.date_range(start_date, end_date)
    selected_dates = sorted(np.random.choice(all_dates, size=work_days, replace=False))
    
    remaining_hours = planned_hours
    for i, log_date in enumerate(selected_dates):
        # random hours worked 4–8 ชั่วโมง
        hours = round(min(np.random.uniform(4, 8), remaining_hours), 2)
        remaining_hours -= hours
        if hours <= 0:
            break

        # random unit_completed 10–30 units
        units = int(np.random.uniform(10, 31))

        log_id = f'LOG{log_counter:04d}'
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

        if remaining_hours <= 0:
            break

    return worklogs, log_counter

# สร้าง worklog สำหรับทุก assignment
worklog_list = []
log_counter = 1
for _, row in assignments.iterrows():
    worklogs, log_counter = generate_worklog(row, log_counter)
    worklog_list.extend(worklogs)

worklog_df = pd.DataFrame(worklog_list)

# Insert batch เข้า DB
from psycopg2.extras import execute_batch
cur = conn.cursor()
insert_query = """
INSERT INTO worklog (log_id, worker_id, project_id, task_id, subtask_id, unit_completed, log_date, hours_worked)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""
data = worklog_df[['log_id','worker_id','project_id','task_id','subtask_id','unit_completed','log_date','hours_worked']].values.tolist()
execute_batch(cur, insert_query, data, page_size=100)
conn.commit()
cur.close()

print(f"Inserted {len(worklog_df)} worklog records.")

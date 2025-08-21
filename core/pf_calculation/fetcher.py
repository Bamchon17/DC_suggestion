# core/pf_calculation/fetcher.py
from db.connection import fetch_query
import pandas as pd

def fetch_assignments(project_id=None) -> pd.DataFrame:
    query = """
        SELECT assignment_id, worker_id, project_id, task_id, subtask_id, planned_hours
        FROM worker_assignment
        WHERE planned_hours IS NOT NULL AND worker_id IS NOT NULL
    """
    if project_id:
        query += f" AND project_id='{project_id}'"
    columns = ['assignment_id', 'worker_id', 'project_id', 'task_id', 'subtask_id', 'planned_hours']
    df = fetch_query(query, columns)
    df['planned_hours'] = pd.to_numeric(df['planned_hours'], errors='coerce').fillna(0)
    df['worker_id'] = df['worker_id'].fillna('unknown')
    return df

def fetch_worklogs(project_id=None) -> pd.DataFrame:
    query = """
        SELECT log_id, worker_id, project_id, task_id, subtask_id, hours_worked, unit_completed
        FROM worklog
        WHERE hours_worked IS NOT NULL AND worker_id IS NOT NULL
    """
    if project_id:
        query += f" AND project_id='{project_id}'"
    columns = ['log_id', 'worker_id', 'project_id', 'task_id', 'subtask_id', 'hours_worked', 'unit_completed']
    df = fetch_query(query, columns)
    if not df.empty:
        df['hours_worked'] = pd.to_numeric(df['hours_worked'], errors='coerce').fillna(0)
        df['unit_completed'] = pd.to_numeric(df['unit_completed'], errors='coerce').fillna(0)
        df['worker_id'] = df['worker_id'].fillna('unknown')
    return df

def fetch_subtasks(project_id=None) -> pd.DataFrame:
    query = """
        SELECT subtask_id, task_id, sub_task_name, qty
        FROM subtask
        WHERE qty IS NOT NULL
    """
    columns = ['subtask_id', 'task_id', 'sub_task_name', 'qty']
    subtasks_df = fetch_query(query, columns)
    if project_id:
        assignments = fetch_assignments(project_id)
        valid_subtasks = assignments[['subtask_id', 'task_id']].drop_duplicates()
        subtasks_df = subtasks_df.merge(valid_subtasks, on=['subtask_id', 'task_id'], how='inner')
    subtasks_df['qty'] = pd.to_numeric(subtasks_df['qty'], errors='coerce').fillna(0)
    return subtasks_df


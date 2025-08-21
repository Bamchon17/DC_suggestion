# core/pf_calculation/updater.py
from db.connection import get_engine
from sqlalchemy import insert, Table, MetaData
import pandas as pd

def update_pf_log(pf_df: pd.DataFrame):
    if pf_df.empty:
        print("No data to insert into pf_log")
        return []

    engine = get_engine()
    metadata = MetaData()
    pf_log = Table('pf_log', metadata, autoload_with=engine)

    updated_ids = []
    with engine.begin() as conn:
        for _, row in pf_df.iterrows():
            worker_id = row['worker_id'] if row['worker_id'] != 'unknown' else (row['worker_ids'][0] if row['worker_ids'] else 'unknown')
            stmt = insert(pf_log).values(
                worker_id=worker_id,
                project_id=row.get('project_id', '12'),
                task_id=row['task_id'],
                subtask_id=row['subtask_id'],
                assignment_id=row['assignment_id'],
                log_date=row['log_date'],
                planned_hours=row['planned_hours'],
                actual_hours=row['hours_worked'],
                unit_completed=row['unit_completed'],
                qty_total=row['qty'],
                pf_time=row['pf_time'],
                pf_qty=row['pf_qty']
            ).returning(pf_log.c.pf_id)

            result = conn.execute(stmt)
            updated_ids.append(result.scalar())

    return updated_ids
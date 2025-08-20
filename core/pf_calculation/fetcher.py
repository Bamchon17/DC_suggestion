# core/pf_calculation/fetcher.py
from db.connection import fetch_query, conn

def fetch_data():
    query = """
    SELECT
        p.project_id,
        p.project_name,
        t.task_id,
        t.task_name,
        st.subtask_id,
        st.sub_task_name,
        st.durations_subtask,
        st.qty,
        st.unit,
        wa.worker_id,
        w.worker_name,
        COALESCE(wl.unit_completed, 0) AS unit_completed,
        ws.standard_rate,
        COUNT(wa.worker_id) OVER (PARTITION BY st.subtask_id) AS workers_assigned
    FROM Subtask st
    JOIN Tasks t ON st.task_id = t.task_id
    JOIN Projects p ON t.project_id = p.project_id
    LEFT JOIN Worker_Assignment wa ON wa.subtask_id = st.subtask_id
    LEFT JOIN Worker w ON w.worker_id = wa.worker_id
    LEFT JOIN Worklog wl ON wl.subtask_id = st.subtask_id AND wl.worker_id = wa.worker_id
    LEFT JOIN Work_standard ws ON ws.task_name = st.sub_task_name
    """
    columns = [
        "project_id","project_name","task_id","task_name",
        "subtask_id","sub_task_name","durations_subtask",
        "qty","unit","worker_id","worker_name",
        "unit_completed","standard_rate","workers_assigned"
    ]
    df = fetch_query(conn, query, columns=columns)
    return df

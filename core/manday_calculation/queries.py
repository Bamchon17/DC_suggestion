# core/manday_calculation/queries.py

# -------------------- Worker Query --------------------
worker_query = """
SELECT 
    w.worker_id,
    w.worker_name,
    st.skill_name,
    sr.skill_level,
    sr.evaluation
FROM worker w
LEFT JOIN skill_record sr ON sr.worker_id = w.worker_id
LEFT JOIN skill_type st ON st.skill_type_id = sr.skill_type_id
"""

# -------------------- Project / Task / Subtask Query --------------------
project_query = """
SELECT 
    p.project_id,
    p.project_name,
    t.task_id,
    t.task_name,
    s.subtask_id,
    s.sub_task_name,
    s.qty,
    s.unit,
    s.start_date,
    s.end_date,
    s.durations_subtask
FROM projects p
LEFT JOIN tasks t ON t.project_id = p.project_id
LEFT JOIN subtask s ON s.task_id = t.task_id
"""

# -------------------- Work Standard Query --------------------
work_standard_query = """
SELECT task_name, standard_rate, unit
FROM work_standard
"""


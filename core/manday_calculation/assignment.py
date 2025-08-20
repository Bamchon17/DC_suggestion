import uuid
import pandas as pd
import numpy as np
from collections import defaultdict
from .utils import normalize_text, best_string_match
from .skill_matching import clean_skill_level
import google.generativeai as genai

def assign_workers(df_tasks: pd.DataFrame, worker_df: pd.DataFrame) -> pd.DataFrame:
    w = worker_df.copy()
    w["skill_name"] = w["skill_name"].map(lambda s: normalize_text(s).lower())
    w = clean_skill_level(w)
    w["evaluation"] = pd.to_numeric(w["evaluation"], errors="coerce")
    w = w.dropna(subset=["evaluation", "skill_name"])  # keep valid

    skill_to_workers: dict[str, list[dict]] = defaultdict(list)
    for _, r in w.iterrows():
        skill_to_workers[r["skill_name"]].append({
            "worker_id": r["worker_id"],
            "worker_name": r["worker_name"],
            "evaluation": r["evaluation"],
            "skill_level": r["skill_level"],
        })

    out_rows = []
    used_workers = set()

    for _, r in df_tasks.iterrows():
        needed = int(r.get("workers_needed", 1)) if not pd.isna(r.get("workers_needed")) else 1
        matched = str(r.get("matched_skill", "")).strip().lower()
        tokens = [t.strip() for t in matched.split(',') if t.strip()] if matched and matched != "ไม่สามารถจับคู่ได้" else []

        candidate_pool = []
        for tok in tokens:
            best = best_string_match(tok, list(skill_to_workers.keys()))
            if best:
                candidate_pool.extend(skill_to_workers[best])
        candidate_pool = sorted(candidate_pool, key=lambda x: (-float(x["evaluation"]) if not pd.isna(x["evaluation"]) else -0.0))

        assigned_count = 0
        if candidate_pool:
            for cand in candidate_pool:
                if cand["worker_name"] in used_workers:
                    continue
                planned_hours = r.get("durations_subtask", 0) * 8 if pd.notna(r.get("durations_subtask")) and r.get("durations_subtask", 0) > 0 else 8
                out_rows.append({
                    "assignment_id": str(uuid.uuid4()),
                    "task_name": r.get("task_name"),
                    "sub_task": r.get("sub_task"),
                    "matched_skill": matched if matched else "ไม่สามารถจับคู่ได้",
                    "start_date": r.get("start_date"),
                    "end_date": r.get("end_date"),
                    "unit": r.get("unit"),
                    "qty": r.get("qty"),
                    "durations_subtask": r.get("durations_subtask"),
                    "standard_rate": r.get("standard_rate"),
                    "workers_needed": needed,
                    "worker_id": cand["worker_id"],
                    "worker_name": cand["worker_name"],
                    "evaluation": cand["evaluation"],
                    "project_id": r.get("project_id"),
                    "task_id": r.get("task_id"),
                    "subtask_id": r.get("subtask_id"),
                    "planned_hours": planned_hours,
                    "assigned_by": "system"
                })
                used_workers.add(cand["worker_name"])
                assigned_count += 1
                if assigned_count >= needed:
                    break
        if assigned_count < needed:
            planned_hours = r.get("durations_subtask", 0) * 8 if pd.notna(r.get("durations_subtask")) and r.get("durations_subtask", 0) > 0 else 8
            out_rows.append({
                "assignment_id": str(uuid.uuid4()),
                "task_name": r.get("task_name"),
                "sub_task": r.get("sub_task"),
                "matched_skill": matched if matched else "ไม่สามารถจับคู่ได้",
                "start_date": r.get("start_date"),
                "end_date": r.get("end_date"),
                "unit": r.get("unit"),
                "qty": r.get("qty"),
                "durations_subtask": r.get("durations_subtask"),
                "standard_rate": r.get("standard_rate"),
                "workers_needed": needed,
                "worker_id": None,
                "worker_name": "ไม่สามารถจับคู่ได้",
                "evaluation": np.nan,
                "project_id": r.get("project_id"),
                "task_id": r.get("task_id"),
                "subtask_id": r.get("subtask_id"),
                "planned_hours": planned_hours,
                "assigned_by": "system"
            })

    df_assigned = pd.DataFrame(out_rows)
    print(f"assign_workers output shape: {df_assigned.shape}, task_id nulls: {df_assigned['task_id'].isna().sum()}, subtask_id nulls: {df_assigned['subtask_id'].isna().sum()}")
    return df_assigned
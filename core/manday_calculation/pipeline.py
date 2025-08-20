import os
import sys
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from .skill_matching import worker_query, project_query, match_skills_to_tasks
from .standard_matching import work_standard_query, match_standards_to_tasks
from .assignment import assign_workers
from .utils import normalize_text, model
from db.connection import dsn, engine, fetch_query  # ตรวจสอบการนำเข้า fetch_query

if __name__ == "__main__":
    print("📥 Loading data from database …")
    worker_df = fetch_query(
        worker_query,  # ส่ง query โดยตรง
        columns=["worker_id", "worker_name", "skill_name", "skill_level", "evaluation"],
    )
    project_df = fetch_query(
        project_query,
        columns=["project_id", "project_name", "task_id", "task_name", "subtask_id", "sub_task_name", "qty", "unit", "start_date", "end_date", "durations_subtask"],
    )
    standard_df = fetch_query(
        work_standard_query,
        columns=["task_name", "standard_rate", "unit"]
    )

    print(f"Loaded project_df with shape: {project_df.shape}, task_id nulls: {project_df['task_id'].isna().sum()}, subtask_id nulls: {project_df['subtask_id'].isna().sum()}")

    project_norm = project_df.copy()
    project_norm["task_name"] = project_norm["task_name"].map(lambda s: normalize_text(s).lower())
    project_norm["sub_task_name"] = project_norm["sub_task_name"].map(lambda s: normalize_text(s).lower())

    print("🔍 Matching skills to tasks with Gemini …")
    df_skills = match_skills_to_tasks(worker_df, project_df, model)
    if df_skills.empty:
        print("⚠️ No skills matched, exiting.")
        sys.exit(1)

    print("🔧 Matching work standards to tasks with Gemini …")
    df_std = match_standards_to_tasks(
        df_skills[["task_name", "sub_task"]].drop_duplicates(),
        standard_df,
        model
    )

    df_merged = df_skills.merge(df_std, on=["task_name", "sub_task"], how="left")
    print(f"df_merged shape: {df_merged.shape}, task_id nulls: {df_merged['task_id'].isna().sum()}, subtask_id nulls: {df_merged['subtask_id'].isna().sum()}")

    df_merged["qty"] = pd.to_numeric(df_merged["qty"], errors="coerce")
    df_merged["standard_rate"] = pd.to_numeric(df_merged["standard_rate"], errors="coerce")
    df_merged["durations_subtask"] = pd.to_numeric(df_merged["durations_subtask"], errors="coerce")
    df_merged["workers_needed"] = df_merged.apply(
        lambda r: int(np.ceil(
            r["qty"] / (r["standard_rate"] * r["durations_subtask"])
        ))
        if pd.notna(r.get("qty"))
           and pd.notna(r.get("standard_rate"))
           and r["standard_rate"] > 0
           and pd.notna(r.get("durations_subtask"))
           and r["durations_subtask"] > 0
        else 1,
        axis=1,
    )

    print("👷 Assigning workers …")
    df_assigned = assign_workers(df_merged, worker_df)

    out1 = os.path.join(os.getcwd(), "matched_skills.csv")
    out2 = os.path.join(os.getcwd(), "matched_standards.csv")
    out3 = os.path.join(os.getcwd(), "assigned_workers.csv")

    df_skills.to_csv(out1, index=False, encoding="utf-8")
    df_std.to_csv(out2, index=False, encoding="utf-8")
    df_assigned.to_csv(out3, index=False, encoding="utf-8")

    print("✅ Done. Saved:")
    print(" -", out1)
    print(" -", out2)
    print(" -", out3)

    worker_ids = df_assigned[df_assigned["worker_id"].notna()]["worker_id"].unique().tolist()

    with engine.begin() as conn_sql:
        if worker_ids:
            conn_sql.execute(
                text("""
                    UPDATE worker_status
                    SET status = 'ไม่ว่าง'
                    WHERE worker_id = ANY(:ids)
                """),
                {"ids": worker_ids}
            )
            print(f"Updated worker_status for {len(worker_ids)} workers to 'ไม่ว่าง'")

    df_to_upload = df_assigned[["assignment_id", "worker_id", "project_id", "task_id", "subtask_id", "start_date", "end_date", "planned_hours", "assigned_by"]].copy()

    df_to_upload["assignment_id"] = df_to_upload["assignment_id"].astype(str)
    df_to_upload["worker_id"] = df_to_upload["worker_id"].astype(str, errors="ignore").replace("nan", None)
    df_to_upload["project_id"] = df_to_upload["project_id"].astype(str, errors="ignore").replace("nan", None)
    df_to_upload["task_id"] = df_to_upload["task_id"].astype(str, errors="ignore").replace("nan", None)
    df_to_upload["subtask_id"] = df_to_upload["subtask_id"].astype(str, errors="ignore").replace("nan", None)
    df_to_upload["start_date"] = pd.to_datetime(df_to_upload["start_date"], errors="coerce")
    df_to_upload["end_date"] = pd.to_datetime(df_to_upload["end_date"], errors="coerce")
    df_to_upload["planned_hours"] = pd.to_numeric(df_to_upload["planned_hours"], errors="coerce")
    df_to_upload["assigned_by"] = df_to_upload["assigned_by"].astype(str)

    null_task_ids = df_to_upload[df_to_upload["task_id"].isna()]
    null_subtask_ids = df_to_upload[df_to_upload["subtask_id"].isna()]
    if not null_task_ids.empty:
        print(f"⚠️ {len(null_task_ids)} rows with null task_id: {null_task_ids[['task_name', 'sub_task']].to_dict('records')}")
    if not null_subtask_ids.empty:
        print(f"⚠️ {len(null_subtask_ids)} rows with null subtask_id: {null_subtask_ids[['task_name', 'sub_task']].to_dict('records')}")

    df_to_upload = df_to_upload.dropna(subset=["project_id", "task_id", "subtask_id"])
    print(f"df_to_upload shape after dropping nulls: {df_to_upload.shape}")

    with engine.begin() as conn_sql:
        valid_task_ids = conn_sql.execute(text("SELECT task_id FROM Tasks")).fetchall()
        valid_task_ids = {row[0] for row in valid_task_ids}
        valid_subtask_ids = conn_sql.execute(text("SELECT subtask_id FROM Subtask")).fetchall()
        valid_subtask_ids = {row[0] for row in valid_subtask_ids}

        invalid_tasks = df_to_upload[~df_to_upload["task_id"].isin(valid_task_ids)]
        invalid_subtasks = df_to_upload[~df_to_upload["subtask_id"].isin(valid_subtask_ids)]
        if not invalid_tasks.empty:
            print(f"⚠️ Invalid task_ids: {invalid_tasks['task_id'].unique().tolist()}")
        if not invalid_subtasks.empty:
            print(f"⚠️ Invalid subtask_ids: {invalid_subtasks['subtask_id'].unique().tolist()}")

        df_to_upload = df_to_upload[df_to_upload["task_id"].isin(valid_task_ids) & df_to_upload["subtask_id"].isin(valid_subtask_ids)]

    try:
        df_to_upload.to_sql(
            name="worker_assignment",
            con=engine,
            if_exists="append",
            index=False
        )
        print(f"✅ Successfully uploaded {len(df_to_upload)} rows to worker_assignment table")
    except Exception as e:
        print(f"❌ Error uploading to worker_assignment: {e}")
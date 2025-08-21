# main.py
from core.pf_calculation.calculate_pf import calculate_daily_pf
from core.pf_calculation.updater import update_pf_log
import pandas as pd

def main(project_id=None):
    print(f"=== คำนวณ PF สำหรับ Project {project_id} ===")
    pf_df = calculate_daily_pf(project_id)
    if pf_df.empty:
        print("No data to insert")
        return

    updated = update_pf_log(pf_df)
    print("✅ PF log ถูกอัปเดตแล้ว")

    alerts = pf_df[pf_df['alert']].copy()
    if not alerts.empty:
        print("\n--- งานที่มี PF < 1 (ต้องระวัง) ---")
        for _, row in alerts.iterrows():
            print(f"- Subtask {row['subtask_id']} (Task {row['task_id']}) | PF = {row['pf_time']:.2f} | คนงาน: {row['worker_ids']} | Planned = {row['planned_hours']}h, Worked = {row['hours_worked']}h")
        alerts.to_csv('pf_alerts.csv', index=False)

if __name__ == "__main__":
    main(project_id="12")




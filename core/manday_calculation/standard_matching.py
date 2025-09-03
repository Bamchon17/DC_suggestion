# core/manday_calculation/standard_matching.py
from core.manday_calculation.utils import normalize_text
import pandas as pd
import google.generativeai as genai

# -------------------- Queries --------------------
work_standard_query = """
SELECT task_name, standard_rate, unit
FROM work_standard
"""
    
# -------------------- Gemini Matching --------------------
def match_standards_to_tasks(project_df: pd.DataFrame, standard_df: pd.DataFrame, model: genai.GenerativeModel) -> pd.DataFrame:
    std_pool = standard_df['task_name'].dropna().astype(str).map(normalize_text).str.lower().unique().tolist()
    if not std_pool:
        print("⚠️ No standards found in standard_df.")
        return pd.DataFrame()

    task_lines = [
        f"- งาน {normalize_text(row['task_name'])}, รายละเอียดงานย่อย {normalize_text(row['sub_task'])}"
        for _, row in project_df.iterrows()
    ]
    
    task_lines_str = '\n'.join(task_lines)

    prompt = f"""
คุณเป็นผู้ช่วยจับคู่งานกับมาตรฐานการทำงานจากรายการต่อไปนี้: {', '.join(std_pool)}
คุณต้องเลือกเฉพาะมาตรฐานที่ใกล้เคียงที่สุดจากรายการ ถ้าไม่ตรงให้เลือกที่คล้ายที่สุด ถ้าไม่พบเลยให้ 'ไม่สามารถจับคู่ได้' ห้ามคิดมาตรฐานใหม่

ตัวอย่าง:
- งาน: การทาสีผนังภายใน, รายละเอียดงานย่อย: การทาสีผนังภายใน -> การทาสี
- งาน: การติดตั้งระบบไฟฟ้า, รายละเอียดงานย่อย: การเดินสายไฟหลัก -> การเดินสายไฟ

ข้อกำหนดรูปแบบคำตอบ:
- ตาราง Markdown 3 คอลัมน์ (ชื่องาน  งานย่อย  มาตรฐานที่จับคู่ได้)
- ถ้าจับคู่ไม่ได้ ให้ใส่ 'ไม่สามารถจับคู่ได้'

รายการงาน:
{(task_lines_str)}

"""
    try:
        resp = model.generate_content(prompt)
        lines = resp.text.split('\n')
    except Exception as e:
        print(f"Error calling Gemini API for standards: {e}")
        return pd.DataFrame()

    rows = []
    header_seen = False
    for raw in lines:
        line = raw.strip()
        if not line or '-' in line:
            continue
        if not header_seen and ('ชื่องาน' in line and 'งานย่อย' in line):
            header_seen = True
            continue
        if '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) != 3:
            continue
        rows.append({
            'task_name': normalize_text(parts[0]).lower(),
            'sub_task': normalize_text(parts[1]).lower(),
            'matched_standard': normalize_text(parts[2]).lower(),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ match_standards_to_tasks returned empty DataFrame")
        return df

    std = standard_df.copy()
    std['task_name'] = std['task_name'].map(lambda s: normalize_text(s).lower())

    df = df.merge(std, how='left', left_on='matched_standard', right_on='task_name', suffixes=('', '_std'))
    df = df.rename(columns={'standard_rate': 'standard_rate', 'unit': 'standard_unit'})
    df.drop(columns=['task_name_std'], inplace=True, errors='ignore')
    print(f"match_standards_to_tasks output shape: {df.shape}")
    return df
# core/manday_calculation/skill_matching.py
import pandas as pd
import google.generativeai as genai
from core.manday_calculation.utils import normalize_text

# -------------------- Queries --------------------
worker_query = """
SELECT  
    w.worker_id,
    w.worker_name,
    st.skill_name,
    sr.skill_level,
    sr.evaluation
FROM worker AS w
JOIN skill_record AS sr ON w.worker_id = sr.worker_id
JOIN skill_type AS st ON st.skill_type_id = sr.skill_type_id
WHERE w.worker_id IN (
    SELECT ws.worker_id
    FROM worker_status AS ws
    WHERE ws.status = 'ว่าง'
)
"""

# -------------------- Gemini Matching --------------------
def match_skills_to_tasks(worker_skills_df: pd.DataFrame, project_df: pd.DataFrame, model: genai.GenerativeModel, temp: float = 0.5) -> pd.DataFrame:
    skills_pool = (
        worker_skills_df['skill_name'].dropna().astype(str).map(normalize_text).str.lower().unique().tolist()
    )
    if not skills_pool:
        print("⚠️ No skills found in worker_skills_df.")
        return pd.DataFrame()

    task_lines = [
        f"- งาน {normalize_text(row['task_name'])}, รายละเอียดงานย่อย {normalize_text(row['sub_task_name'])}, task_id {row['task_id']}, subtask_id {row['subtask_id']}"
        for _, row in project_df.iterrows()
    ]
    if not task_lines:
        print("⚠️ No tasks found in project_df.")
        return pd.DataFrame()

    prompt = f"""
คุณเป็นผู้เชี่ยวชาญด้านการก่อสร้าง มีหน้าที่จับคู่ ทักษะแรงงาน กับ งานก่อสร้าง โดยต้องพิจารณา ทั้ง งานหลัก และ งานย่อย ร่วมกัน  
เลือกทักษะที่เหมาะสมที่สุดจากรายการต่อไปนี้เท่านั้น: {', '.join(skills_pool)}

## กติกา
1. พิจารณาทั้ง งานหลัก (task_name) และ รายละเอียดงานย่อย (sub_task) ว่าต้องใช้ทักษะอะไร
2. เลือกเฉพาะทักษะที่มีอยู่ในรายการข้างต้น ห้ามสร้างหรือปรับแต่งทักษะใหม่
3. ถ้าไม่มีทักษะที่ตรง ให้ตอบ 'ไม่สามารถจับคู่ได้'
4. ถ้ามีมากกว่า 1 ทักษะที่เหมาะสม ให้คั่นด้วย ', '
5. ห้ามเลือกทักษะที่ไม่เกี่ยวข้องกับงานหลักและงานย่อยนั้น

## ตัวอย่าง
- งานหลัก: งานติดตั้งระบบไฟฟ้า, รายละเอียดงานย่อย: เดินสายไฟภายในตู้  
  → เลือก: ช่างไฟฟ้า, งานระบบไฟฟ้า
- งานหลัก: งานโครงสร้าง, รายละเอียดงานย่อย: เทคอนกรีต  
  → เลือก: ช่างโครงสร้าง, งานคอนกรีต
- งานหลัก: งานตกแต่ง, รายละเอียดงานย่อย: ติดตั้งฝ้าเพดาน  
  → เลือก: ช่างฝ้า, งานตกแต่ง
- งานหลัก: งานถนนคอนกรีต, รายละเอียดงานย่อย: เทคอนกรีตพื้นถนน  
  → ถ้าไม่มีทักษะที่ตรงกับงานนี้ในรายการ ให้ตอบ 'ไม่สามารถจับคู่ได้'

## รูปแบบคำตอบ
- ส่งข้อมูลเป็นตาราง Markdown 5 คอลัมน์ (ชื่องาน  งานย่อย  ทักษะที่จับคู่ได้  task_id  subtask_id)
- ถ้ามีมากกว่า 1 ทักษะ คั่นด้วยเครื่องหมายจุลภาคและช่องว่าง
- ถ้าจับคู่ไม่ได้ ให้ใส่ 'ไม่สามารถจับคู่ได้'
- ต้องคงค่า task_id และ subtask_id จากข้อมูลที่ให้มา

รายการงานที่ต้องการจับคู่:
{chr(10).join(task_lines)}
"""
    
    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temp,
                top_p=0.9,
                top_k=40
            )
        )
        lines = resp.text.split('\n')
    except Exception as e:
        print(f"Error calling Gemini API for skills: {e}")
        return pd.DataFrame()
    
    results = []
    header_seen = False
    for raw in lines:
        line = raw.strip()
        if not line or '-' in line:
            continue
        if not header_seen and ('ชื่องาน' in line and 'งานย่อย' in line and 'task_id' in line and 'subtask_id' in line):
            header_seen = True
            continue
        if '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) != 5:
            continue
        task_name, sub_task, skill_raw, task_id, subtask_id = parts
        task_name = normalize_text(task_name).lower()
        sub_task = normalize_text(sub_task).lower()
        results.append({
            'task_name': task_name,
            'sub_task': sub_task,
            'matched_skill': skill_raw if skill_raw.strip() else 'ไม่สามารถจับคู่ได้',
            'project_id': project_df[project_df['task_name'].map(normalize_text).str.lower() == task_name]['project_id'].iloc[0] if not project_df[project_df['task_name'].map(normalize_text).str.lower() == task_name].empty else None,
            'task_id': task_id if task_id else None,
            'subtask_id': subtask_id if subtask_id else None,
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("⚠️ match_skills_to_tasks returned empty DataFrame")
        return df

    pj = project_df.copy()
    pj['task_name'] = pj['task_name'].map(lambda s: normalize_text(s).lower())
    pj['sub_task_name'] = pj['sub_task_name'].map(lambda s: normalize_text(s).lower())

    df = df.merge(
        pj[['task_name', 'sub_task_name', 'qty', 'unit', 'start_date', 'end_date', 'durations_subtask', 'project_id', 'task_id', 'subtask_id']],
        how='left',
        left_on=['task_name', 'sub_task', 'project_id'],
        right_on=['task_name', 'sub_task_name', 'project_id'],
        suffixes=('', '_pj')
    )

    df['task_id'] = df.apply(
        lambda row: row['task_id_pj'] if pd.isna(row['task_id']) and not pd.isna(row['task_id_pj']) else row['task_id'],
        axis=1
    )
    df['subtask_id'] = df.apply(
        lambda row: row['subtask_id_pj'] if pd.isna(row['subtask_id']) and not pd.isna(row['subtask_id_pj']) else row['subtask_id'],
        axis=1
    )

    df = df.drop(columns=['sub_task_name', 'task_id_pj', 'subtask_id_pj'], errors='ignore')
    print(f"match_skills_to_tasks output shape: {df.shape}, task_id nulls: {df['task_id'].isna().sum()}, subtask_id nulls: {df['subtask_id'].isna().sum()}")
    return df
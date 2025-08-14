# Labour Management System

โปรเจกต์นี้เป็นระบบจัดการแรงงานก่อสร้าง (Construction Labour Management) สำหรับช่วยคำนวณ **manday**, จัดสรรคนงานตาม **skill**, ติดตามงาน และประเมิน **Productivity Factor (PF)** ของงานก่อสร้างในแต่ละโครงการ  

ระบบนี้ใช้ **Python**, **PostgreSQL**, และ **VSCode** เป็นหลัก โดยมีโครงสร้างไฟล์เพื่อแยก logic ของฝั่งต่าง ๆ อย่างชัดเจน

---

## Features

1. **Project, Task, Subtask Management**
   - จัดการโครงการ งานหลัก และงานย่อย
   - เก็บข้อมูล contract quantity, durations, start/end date, และ status

2. **Worker & Skill Management**
   - เก็บข้อมูลแรงงาน, skill, availability และ limitation
   - LLM mapping skill กับ subtask

3. **Manpower Calculation**
   - คำนวณจำนวนคนงาน (manday) ตาม standard rate และ unit ของแต่ละงาน

4. **Work Assignment**
   - จัดสรรคนงานลง subtask ตาม manday
   - รองรับ update worker_assignment

5. **Productivity Factor (PF) Calculation**
   - ติดตามความคืบหน้าและประสิทธิภาพงาน
   - PF = actual_rate / estimated_rate

6. **Delayed Task Assignment**
   - หาก PF < 1 จะหาคนงานที่ว่างและตรง skill มาช่วย

---

## Project Structure
project_labour_management/
│
├── data/                        # เก็บไฟล์ CSV, Excel, หรือ mock data
│   ├── tasks.csv
│   ├── workers.csv
│   ├── worklog.csv
│   ├── work_standard.csv
│   └── other_reference_data/
│
├── db/                          # Scripts ที่เกี่ยวกับ database
│   └── connection.py             # ตัวเชื่อมกับ PostgreSQL
│   
│
├── core/                        # Logic หลัก
│   ├── __init__.py
│   ├── manday_calculation/       # Tae
│   │   ├── calculate_manday.py
│   │   └── assign_workers.py
│   ├── pf_calculation/           # Bam
│   │   ├── calculate_pf.py
│   │   └── analytics.py
│   ├── skill_matching/
│   │   ├── map_skills.py
│   │   └── embedding_utils.py
│   └── utils.py                  # ฟังก์ชันช่วยเหลือทั่วไป เช่น date utils, logging
│
├── ui/                          # Frontend หรือ Streamlit / web app
│   ├── app.py                    # Main UI
│   ├── pages/                    # หน้าแยกตาม module
│   │   ├── project_overview.py
│   │   ├── task_assignment.py
│   │   └── pf_dashboard.py
│   └── components/               # Reusable components
│       ├── tables.py
│       └── charts.py
│
├── tests/                        # Unit test และ integration test
│   ├── test_manday.py
│   ├── test_pf.py
│   └── test_skill_matching.py
│
├── notebooks/                    # Jupyter notebooks สำหรับ exploration หรือ debugging
│   ├── pf_analysis.ipynb
│   └── manday_demo.ipynb
│
├── requirements.txt              # Dependencies
├── .env                          # เก็บ DB_HOST, DB_USER, DB_PASSWORD, etc.
└── README.md


git clone https://github.com/Bamchon17/DC_suggestion.git
cd DC_suggestion

# สร้าง virtualenv
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# ติดตั้ง dependency
pip install -r requirements.txt

# run

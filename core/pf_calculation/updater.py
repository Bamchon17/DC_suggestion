# core/pf_calculation/update_log.py
from .fetcher import fetch_data
from .calculate_pf import calculate_pf
from db.connection import engine

def update_pf_log():
    df = fetch_data()
    pf_log_df = calculate_pf(df)
    if not pf_log_df.empty:
        pf_log_df.to_sql('pf_log', con=engine, if_exists='append', index=False)
        print(f"✅ PF log updated: {len(pf_log_df)} rows")

if __name__ == "__main__":
    update_pf_log()

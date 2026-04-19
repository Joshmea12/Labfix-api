from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from typing import Optional

SUPABASE_URL = "https://zrtlmgrjrhrgmzraezbk.supabase.co"
SUPABASE_KEY = "sb_publishable_iw6hKMWFcnXjFqfr2UapRw_xWDTD31r"

db = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Log(BaseModel):
    machine: str
    symptom: str
    fault: str
    notes: Optional[str] = ""
    technician: Optional[str] = "Unknown"

@app.get("/")
def root():
    return {"status": "LabFix API is live"}

@app.post("/log")
def save_log(log: Log):
    db.table("repair_logs").insert(log.dict()).execute()
    return {"success": True}

@app.get("/logs")
def get_logs():
    result = db.table("repair_logs").select("*").order("created_at", desc=True).execute()
    return result.data

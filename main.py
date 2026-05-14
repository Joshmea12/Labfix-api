import os
import hmac
import hashlib
import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from supabase import create_client, Client

# ── LOGGING ──────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("labfix")

# ── ENV VARIABLES ────────────────────────────
SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_API_KEY       = os.environ["GEMINI_API_KEY"]
PAYSTACK_SECRET_KEY  = os.environ["PAYSTACK_SECRET_KEY"]

# ── CLIENTS ──────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# ── SYSTEM PROMPT ─────────────────────────────
SYSTEM_PROMPT = """
You are LabFix AI, a specialized laboratory equipment diagnostic and maintenance assistant.
You serve lab technicians, biomedical engineers, and research scientists worldwide.

STRICT RULES:
1. NEVER fabricate or guess a diagnosis. If you cannot confidently identify the equipment
   or fault, respond: "I cannot make a confident diagnosis from the information provided.
   Please provide a clearer image or more specific details."
2. Always state confidence level: High / Medium / Low.
3. For every valid diagnosis include:
   - Equipment identified (full name and type)
   - Observed fault or symptom
   - Most likely root cause with technical reasoning
   - Recommended repair steps (numbered and specific)
   - Parts likely needed
   - Safety warning if electrical, chemical, or pressure risk exists
4. If image is blurry or does not show lab equipment respond:
   "This image does not show recognizable laboratory equipment or image quality
   is too low for accurate diagnosis. Please retake the photo."
5. Do NOT answer anything outside laboratory equipment maintenance. Say:
   "LabFix AI is specialized for laboratory equipment diagnosis only."
6. Base all repair steps on standard engineering practice. Never assume.
"""

# ── APP ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LabFix AI backend starting...")
    yield
    logger.info("LabFix AI backend shutting down...")

app = FastAPI(title="LabFix AI API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://repairwiz-ai-core.lovable.app",
        "https://labfix-ai-core.lovable.app",
        "https://labfixai.lovable.app",
        "https://labfix-ai.lovable.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ── AUTH ──────────────────────────────────────
async def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.replace("Bearer ", "")
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user_response.user
    except Exception as e:
        logger.warning(f"Auth error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# ── MODELS ────────────────────────────────────
class DiagnoseRequest(BaseModel):
    image_base64: str
    symptom: Optional[str] = "No symptom described"
    media_type: Optional[str] = "image/jpeg"

    @validator("image_base64")
    def validate_image(cls, v):
        if not v or len(v) < 100:
            raise ValueError("Invalid image data")
        return v

    @validator("media_type")
    def validate_media_type(cls, v):
        allowed = ["image/jpeg", "image/png", "image/webp", "image/gif"]
        if v not in allowed:
            raise ValueError(f"Unsupported media type. Use: {allowed}")
        return v

class RepairLogRequest(BaseModel):
    machine: str
    symptom: str
    fault: str
    notes: Optional[str] = ""
    technician: str

    @validator("machine", "symptom", "fault", "technician")
    def sanitize_fields(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Field cannot be empty")
        if len(v) > 500:
            raise ValueError("Field too long")
        return v.strip()

class MachineRequest(BaseModel):
    name: str
    category: str
    icon: Optional[str] = "🔬"

    @validator("name", "category")
    def sanitize_fields(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Field cannot be empty")
        return v.strip()

class SearchRequest(BaseModel):
    query: str

    @validator("query")
    def sanitize_query(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Search query too short")
        if len(v) > 300:
            raise ValueError("Search query too long")
        return v.strip()

# ── HEALTH CHECK ──────────────────────────────
@app.get("/")
async def health_check():
    return {
        "status": "LabFix AI is running",
        "version": "2.0.0",
        "powered_by": "Gemini AI + Supabase + Paystack"
    }

# ── DIAGNOSE ──────────────────────────────────
@app.post("/diagnose")
async def diagnose_equipment(body: DiagnoseRequest, user=Depends(get_current_user)):
    try:
        usage = supabase.rpc(
            "increment_diagnosis_count",
            {"user_uuid": str(user.id)}
        ).execute()

        result = usage.data
        if not result.get("allowed"):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "free_limit_reached",
                    "message": "You have used all 5 free diagnoses this month. Upgrade to Pro for unlimited access.",
                    "count": result.get("count"),
                    "plan": result.get("plan")
                }
            )

        try:
            image_data = base64.b64decode(body.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        prompt = f"{SYSTEM_PROMPT}\n\nAnalyze this laboratory equipment image carefully.\nSymptom reported: {body.symptom}\n\nProvide a full diagnostic report."

        response = gemini_model.generate_content([
            {"mime_type": body.media_type, "data": image_data},
            prompt
        ])

        diagnosis_text = response.text if response.text else (
            "I cannot make a confident diagnosis from the information provided. "
            "Please provide a clearer image or more specific details."
        )

        return {
            "success": True,
            "diagnosis": diagnosis_text,
            "usage": {
                "count": result.get("count"),
                "plan": result.get("plan"),
                "remaining": max(0, 5 - result.get("count", 0)) if result.get("plan") == "free" else "unlimited"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Diagnosis error for user {user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Diagnosis service temporarily unavailable")

# ── SEARCH ────────────────────────────────────
@app.post("/search")
async def search_equipment(body: SearchRequest, user=Depends(get_current_user)):
    try:
        prompt = f"{SYSTEM_PROMPT}\n\nA lab technician is asking: {body.query}\n\nIf related to laboratory equipment provide helpful guidance. If unrelated, decline per your rules."
        response = gemini_model.generate_content(prompt)
        return {
            "success": True,
            "result": response.text or "No results found for your query."
        }
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Search service temporarily unavailable")

# ── REPAIR LOGS ───────────────────────────────
@app.get("/repair-logs")
async def get_repair_logs(user=Depends(get_current_user)):
    try:
        logs = supabase.table("repair_logs") \
            .select("*") \
            .eq("user_id", str(user.id)) \
            .order("created_at", desc=True) \
            .execute()
        return {"success": True, "data": logs.data}
    except Exception as e:
        logger.error(f"Fetch logs error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch repair logs")

@app.post("/repair-logs")
async def create_repair_log(body: RepairLogRequest, user=Depends(get_current_user)):
    try:
        log = supabase.table("repair_logs").insert({
            "machine": body.machine,
            "symptom": body.symptom,
            "fault": body.fault,
            "notes": body.notes,
            "technician": body.technician,
            "user_id": str(user.id)
        }).execute()
        return {"success": True, "data": log.data}
    except Exception as e:
        logger.error(f"Create log error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create repair log")

@app.delete("/repair-logs/{log_id}")
async def delete_repair_log(log_id: str, user=Depends(get_current_user)):
    try:
        supabase.table("repair_logs") \
            .delete() \
            .eq("id", log_id) \
            .eq("user_id", str(user.id)) \
            .execute()
        return {"success": True, "message": "Repair log deleted"}
    except Exception as e:
        logger.error(f"Delete log error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not delete repair log")

# ── MACHINES ──────────────────────────────────
@app.get("/machines")
async def get_machines(user=Depends(get_current_user)):
    try:
        machines = supabase.table("Machines").select("*").order("name").execute()
        return {"success": True, "data": machines.data}
    except Exception as e:
        logger.error(f"Fetch machines error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch machines")

@app.post("/machines")
async def create_machine(body: MachineRequest, user=Depends(get_current_user)):
    try:
        machine = supabase.table("Machines").insert({
            "name": body.name,
            "category": body.category,
            "icon": body.icon,
            "user_id": str(user.id)
        }).execute()
        return {"success": True, "data": machine.data}
    except Exception as e:
        logger.error(f"Create machine error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create machine")

# ── PROFILE ───────────────────────────────────
@app.get("/profile")
async def get_profile(user=Depends(get_current_user)):
    try:
        profile = supabase.table("profiles") \
            .select("*") \
            .eq("id", str(user.id)) \
            .single() \
            .execute()
        return {"success": True, "data": profile.data}
    except Exception as e:
        logger.error(f"Fetch profile error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch profile")

# ── PAYSTACK WEBHOOK ──────────────────────────
@app.post("/webhook/paystack")
async def paystack_webhook(request: Request):
    paystack_signature = request.headers.get("x-paystack-signature")
    if not paystack_signature:
        raise HTTPException(status_code=400, detail="Missing Paystack signature")

    body_bytes = await request.body()

    expected_signature = hmac.new(
        key=PAYSTACK_SECRET_KEY.encode("utf-8"),
        msg=body_bytes,
        digestmod=hashlib.sha512
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, paystack_signature):
        logger.warning("Invalid Paystack webhook signature")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    event = payload.get("event")
    logger.info(f"Paystack event: {event}")

    if event == "charge.success":
        data = payload["data"]
        reference = data["reference"]
        email = data["customer"]["email"]
        amount = data["amount"]

        if amount == 900:
            plan = "pro"
        elif amount == 2900:
            plan = "enterprise"
        else:
            return {"status": "ignored"}

        async with httpx.AsyncClient() as client:
            verify_response = await client.get(
                f"https://api.paystack.co/transaction/verify/{reference}",
                headers={"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"}
            )
            verify_data = verify_response.json()

        if verify_data.get("data", {}).get("status") != "success":
            logger.warning(f"Paystack verification failed: {reference}")
            return {"status": "verification_failed"}

        try:
            users = supabase.auth.admin.list_users()
            user = next((u for u in users if u.email == email), None)
            if not user:
                return {"status": "user_not_found"}

            supabase.rpc("upgrade_user_plan", {
                "user_uuid": str(user.id),
                "new_plan": plan,
                "paystack_ref": reference
            }).execute()

            logger.info(f"User {email} upgraded to {plan}")
            return {"status": "success", "plan": plan}

        except Exception as e:
            logger.error(f"Plan upgrade error: {str(e)}")
            return {"status": "error"}

    return {"status": "ok"}

# ── GLOBAL ERROR HANDLER ──────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong. Please try again."}
    )

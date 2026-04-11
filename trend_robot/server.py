"""
Trend Robot - REST API Server

HEMA platformasi bilan integratsiya uchun FastAPI server.
RSI/Hedge/Sure-Fire bot pattern ga mos — bir xil endpoint struktura.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import psutil
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Header, Request, Depends, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .session_manager import get_session_manager, SessionStatus
from .config import validate_trading_settings

# ═══════════════════════════════════════════════════════════════════════════════
#                               LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class JSONLogFormatter(logging.Formatter):
    """Structured JSON log formatlash"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
        return json.dumps(log_data, ensure_ascii=False)


def _setup_logging():
    """Logging sozlash"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    if root_logger.handlers:
        return
    handler = logging.StreamHandler()
    if LOG_FORMAT == "json":
        handler.setFormatter(JSONLogFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    root_logger.addHandler(handler)


_setup_logging()
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#                               CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BOT_ID = os.getenv("BOT_ID", "trend-v1")
BOT_NAME = os.getenv("BOT_NAME", "Trend Robot")
BOT_VERSION = os.getenv("BOT_VERSION", "1.0.0")
BOT_SECRET = os.getenv("BOT_SECRET", "")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
ALLOW_INSECURE = os.getenv("ALLOW_INSECURE", "false").lower() in ("true", "1")
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")

# Supported pairs — Bitget USDT-M Futures
SUPPORTED_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT",
    "BNBUSDT", "ATOMUSDT", "NEARUSDT", "OPUSDT", "ARBUSDT",
    "APTUSDT", "SUIUSDT", "PEPEUSDT", "SHIBUSDT", "WIFUSDT",
]

_start_time = time.time()
_supported_pairs_cache: Optional[list] = None
_supported_pairs_cache_time: float = 0

# ═══════════════════════════════════════════════════════════════════════════════
#                               AUTH
# ═══════════════════════════════════════════════════════════════════════════════


def verify_signature(payload: str, timestamp: str, signature: str) -> bool:
    """HMAC-SHA256 imzo tekshirish"""
    if not BOT_SECRET:
        return False
    message = f"{timestamp}.{payload}"
    expected = hmac.new(
        BOT_SECRET.encode(), message.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def verify_request(request: Request):
    """Request autentifikatsiya — HEMA dan kelgan so'rovlarni tekshirish"""
    if ALLOW_INSECURE:
        return

    if not BOT_SECRET:
        raise HTTPException(status_code=503, detail="BOT_SECRET sozlanmagan")

    timestamp = request.headers.get("X-Webhook-Timestamp", "")
    signature = request.headers.get("X-Webhook-Signature", "")

    if not timestamp or not signature:
        raise HTTPException(status_code=401, detail="Auth headerlar yo'q")

    # Timestamp tekshirish (5 daqiqa oyna)
    try:
        ts = int(timestamp) / 1000 if len(timestamp) > 10 else int(timestamp)
        if abs(time.time() - ts) > 300:
            raise HTTPException(status_code=401, detail="Timestamp eskirgan")
    except ValueError:
        raise HTTPException(status_code=401, detail="Noto'g'ri timestamp")

    # Body o'qish
    body = await request.body()
    payload = body.decode("utf-8") if body else ""

    if not verify_signature(payload, timestamp, signature):
        raise HTTPException(status_code=401, detail="Noto'g'ri imzo")


async def verify_admin(x_admin_key: str = Header(None)):
    """Admin endpoint autentifikatsiya"""
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="ADMIN_API_KEY sozlanmagan")
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin kaliti noto'g'ri")


# ═══════════════════════════════════════════════════════════════════════════════
#                               ERROR HANDLER
# ═══════════════════════════════════════════════════════════════════════════════


ERROR_CODE_MAP = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


# ═══════════════════════════════════════════════════════════════════════════════
#                               CONFIG SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG_SCHEMA = {
    "settings": [
        # Grid sozlamalari
        {
            "key": "nPositions", "label": "Grid Positions", "type": "number",
            "default": 5, "min": 1, "max": 50, "step": 1,
            "description": "Maksimal ochiq pozitsiyalar soni (har tomon uchun)",
            "group": "Grid"
        },
        {
            "key": "gridSpacingPct", "label": "Grid Spacing %", "type": "number",
            "default": 0.5, "min": 0.01, "max": 10.0, "step": 0.01,
            "description": "Grid orasidagi masofa (narxning %)",
            "group": "Grid"
        },
        {
            "key": "gridSpacingVolatilityWeight", "label": "Volatility Weight", "type": "number",
            "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1,
            "description": "ATR volatilite ta'siri grid spacingga (0=off)",
            "group": "Grid"
        },
        # Entry sozlamalari
        {
            "key": "emaSpan0", "label": "EMA Span Short", "type": "number",
            "default": 100, "min": 1, "max": 10000, "step": 1,
            "description": "Qisqa EMA davri (shamchalar soni)",
            "group": "Entry"
        },
        {
            "key": "emaSpan1", "label": "EMA Span Long", "type": "number",
            "default": 1000, "min": 1, "max": 10000, "step": 1,
            "description": "Uzun EMA davri (shamchalar soni)",
            "group": "Entry"
        },
        {
            "key": "initialEmaDist", "label": "Initial EMA Distance %", "type": "number",
            "default": 0.3, "min": 0.01, "max": 10.0, "step": 0.01,
            "description": "Birinchi entry uchun EMA dan masofa (%)",
            "group": "Entry"
        },
        {
            "key": "initialQtyPct", "label": "Initial Qty %", "type": "number",
            "default": 1, "min": 0.1, "max": 100, "step": 0.1,
            "description": "Birinchi order hajmi (WEL ning %)",
            "group": "Entry"
        },
        {
            "key": "doubleDownFactor", "label": "Double Down Factor", "type": "number",
            "default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1,
            "description": "Keyingi entry multiplikatoru (0.7 = 70% ko'proq)",
            "group": "Entry"
        },
        # Exit sozlamalari
        {
            "key": "closeGridMarkupStart", "label": "Close Markup Start %", "type": "number",
            "default": 0.6, "min": 0.01, "max": 10.0, "step": 0.01,
            "description": "Birinchi close uchun profit markup (%)",
            "group": "Exit"
        },
        {
            "key": "closeGridMarkupEnd", "label": "Close Markup End %", "type": "number",
            "default": 0.9, "min": 0.01, "max": 20.0, "step": 0.01,
            "description": "Oxirgi close uchun profit markup (%)",
            "group": "Exit"
        },
        {
            "key": "closeGridQtyPct", "label": "Close Qty %", "type": "number",
            "default": 50, "min": 10, "max": 100, "step": 5,
            "description": "Har close da pozitsiyaning necha % yopiladi",
            "group": "Exit"
        },
        # Trailing sozlamalari
        {
            "key": "entryTrailingThreshold", "label": "Entry Trail Threshold %", "type": "number",
            "default": 1.0, "min": 0, "max": 10.0, "step": 0.1,
            "description": "Entry uchun narx harakati threshold (0=off)",
            "group": "Trailing"
        },
        {
            "key": "entryTrailingRetracement", "label": "Entry Trail Retrace %", "type": "number",
            "default": 0.5, "min": 0, "max": 5.0, "step": 0.1,
            "description": "Entry uchun retrace talabi",
            "group": "Trailing"
        },
        {
            "key": "trailingGridRatio", "label": "Trailing/Grid Ratio", "type": "number",
            "default": 0.5, "min": 0, "max": 1.0, "step": 0.1,
            "description": "0=faqat grid, 1=faqat trailing, 0.5=aralash",
            "group": "Trailing"
        },
        # Risk sozlamalari
        {
            "key": "walletExposureLimit", "label": "Wallet Exposure Limit", "type": "number",
            "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
            "description": "Maksimal pozitsiya / balans nisbati",
            "group": "Risk"
        },
        {
            "key": "maxLossPercent", "label": "Max Loss %", "type": "number",
            "default": 20, "min": 0, "max": 100, "step": 1,
            "description": "Umumiy maksimal zarar (0=off)",
            "group": "Risk"
        },
        {
            "key": "hslRedThreshold", "label": "HSL Red Threshold", "type": "number",
            "default": 50, "min": 5, "max": 100, "step": 5,
            "description": "Panic close uchun drawdown % (HSL Red tier)",
            "group": "Risk"
        },
        # Unstuck sozlamalari
        {
            "key": "unstuckThreshold", "label": "Unstuck Threshold %", "type": "number",
            "default": 3, "min": 0.1, "max": 50, "step": 0.1,
            "description": "Necha % zararda stuck hisoblanadi",
            "group": "Unstuck"
        },
        {
            "key": "unstuckClosePct", "label": "Unstuck Close %", "type": "number",
            "default": 5, "min": 1, "max": 100, "step": 1,
            "description": "Har unstuck da pozitsiyaning necha % yopiladi",
            "group": "Unstuck"
        },
        # Fee sozlamalari
        {
            "key": "feeRate", "label": "Fee Rate %", "type": "number",
            "default": 0.1, "min": 0, "max": 1.0, "step": 0.01,
            "description": "Taker fee rate (per side, %)",
            "group": "Fee Settings"
        },
        # Umumiy
        {
            "key": "marginMode", "label": "Margin Mode", "type": "select",
            "default": "crossed", "options": ["crossed", "isolated"],
            "description": "Margin rejimi",
            "group": "General"
        },
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
#                               LIFESPAN
# ═══════════════════════════════════════════════════════════════════════════════

_cleanup_task: Optional[asyncio.Task] = None


async def _cleanup_loop():
    """Eski sessionlarni tozalash (har 1 soatda)"""
    while True:
        try:
            await asyncio.sleep(3600)
            sm = get_session_manager()
            count = await sm.cleanup_old_sessions(max_age_hours=24)
            if count > 0:
                logger.info(f"Tozalandi: {count} ta eski session")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup xatosi: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan — startup va shutdown"""
    global _cleanup_task
    logger.info(f"Starting {BOT_NAME} v{BOT_VERSION} (ID: {BOT_ID})")

    # Cleanup task
    _cleanup_task = asyncio.create_task(_cleanup_loop())

    yield

    # Shutdown
    logger.info("Shutting down...")
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass

    # Barcha sessionlarni to'xtatish
    sm = get_session_manager()
    await sm.stop_all()
    logger.info("Shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
#                               FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=BOT_NAME,
    version=BOT_VERSION,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handler — HEMA format
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code = ERROR_CODE_MAP.get(exc.status_code, "UNKNOWN_ERROR")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": str(exc.detail)}},
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                               HEALTH & INFO
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
@app.get("/api/v1/health")
async def health():
    """Health check"""
    sm = get_session_manager()
    process = psutil.Process()

    return {
        "status": "ok",
        "bot_id": BOT_ID,
        "bot_name": BOT_NAME,
        "version": BOT_VERSION,
        "uptime_seconds": int(time.time() - _start_time),
        "active_sessions": sm.active_count,
        "memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
        "threads": process.num_threads(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/info")
@app.get("/api/v1/info")
async def info():
    """Bot ma'lumotlari va configSchema — HEMA UI uchun"""
    return {
        "id": BOT_ID,
        "name": BOT_NAME,
        "version": BOT_VERSION,
        "description": "Passivbot ilhomlangan contrarian market-maker. "
                       "EMA anchor asosida grid limit orderlar qo'yadi. "
                       "Trailing retrace, unstucking, HSL tiered stop.",
        "strategy": "GRID",
        "mode": "AUTOMATIC",
        "supportedPairs": SUPPORTED_PAIRS,
        "supportedExchanges": ["bitget"],
        "minCapital": 50,
        "recommendedCapital": 500,
        "riskLevel": "MEDIUM",
        "configSchema": CONFIG_SCHEMA,
        "features": [
            "grid_trading", "trailing_entry", "trailing_exit",
            "unstucking", "hsl_tiered_stop", "wallet_exposure_limit",
            "volatility_adaptive_grid", "fee_aware",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                           USER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/users", dependencies=[Depends(verify_request)])
async def register_user(request: Request):
    """Foydalanuvchi ro'yxatdan o'tkazish"""
    body = await request.json()
    sm = get_session_manager()

    user_id = body.get("userId", "")
    if not user_id:
        raise HTTPException(status_code=400, detail="userId majburiy")

    exchange = body.get("exchange", {})
    if not exchange.get("apiKey") or not exchange.get("apiSecret"):
        raise HTTPException(status_code=400, detail="Exchange credentials majburiy")

    settings = body.get("settings", {})
    custom_settings = body.get("customSettings", {})

    # Validate custom settings
    if custom_settings:
        error = validate_trading_settings(custom_settings)
        if error:
            raise HTTPException(status_code=422, detail=error)

    try:
        session_key = await sm.register_user(
            user_id=user_id,
            user_bot_id=body.get("userBotId", user_id),
            exchange=exchange,
            settings=settings,
            custom_settings=custom_settings,
            webhook_url=body.get("webhookUrl", ""),
            webhook_secret=body.get("webhookSecret", ""),
        )
        return {"status": "registered", "sessionKey": session_key}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Register xatosi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/users/{user_id}/start", dependencies=[Depends(verify_request)])
async def start_trading(user_id: str):
    """Trading boshlash"""
    sm = get_session_manager()
    try:
        await sm.start_session(user_id)
        return {"status": "started"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Start xatosi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/users/{user_id}/stop", dependencies=[Depends(verify_request)])
async def stop_trading(user_id: str):
    """Trading to'xtatish"""
    sm = get_session_manager()
    try:
        await sm.stop_session(user_id)
        return {"status": "stopped"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/users/{user_id}/pause", dependencies=[Depends(verify_request)])
async def pause_trading(user_id: str):
    """Trading pauza"""
    sm = get_session_manager()
    try:
        await sm.pause_session(user_id)
        return {"status": "paused"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/users/{user_id}/resume", dependencies=[Depends(verify_request)])
async def resume_trading(user_id: str):
    """Trading davom ettirish"""
    sm = get_session_manager()
    try:
        await sm.resume_session(user_id)
        return {"status": "resumed"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/users/{user_id}/status", dependencies=[Depends(verify_request)])
async def get_status(user_id: str):
    """Foydalanuvchi holati"""
    sm = get_session_manager()
    try:
        status = await sm.get_status(user_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/users/{user_id}/settings", dependencies=[Depends(verify_request)])
async def get_settings(user_id: str):
    """Foydalanuvchi sozlamalari"""
    sm = get_session_manager()
    try:
        settings = await sm.get_settings(user_id)
        return settings
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put("/api/v1/users/{user_id}/config", dependencies=[Depends(verify_request)])
async def update_config(user_id: str, request: Request):
    """Sozlamalarni yangilash (live)"""
    body = await request.json()
    custom_settings = body.get("customSettings", body)

    if custom_settings:
        error = validate_trading_settings(custom_settings)
        if error:
            raise HTTPException(status_code=422, detail=error)

    sm = get_session_manager()
    try:
        await sm.update_settings(user_id, custom_settings)
        return {"status": "updated"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/users/{user_id}/close-positions", dependencies=[Depends(verify_request)])
async def close_positions(user_id: str):
    """Barcha pozitsiyalarni yopish (botni to'xtatmasdan)"""
    sm = get_session_manager()
    try:
        await sm.close_positions(user_id)
        return {"status": "closing"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/users/{user_id}/force-sync", dependencies=[Depends(verify_request)])
async def force_sync(user_id: str):
    """Exchange bilan pozitsiyalarni sinxronlashtirish"""
    sm = get_session_manager()
    try:
        await sm.force_sync(user_id)
        return {"status": "synced"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/v1/users/{user_id}", dependencies=[Depends(verify_request)])
async def delete_user(user_id: str):
    """Foydalanuvchini o'chirish"""
    sm = get_session_manager()
    try:
        await sm.unregister_user(user_id)
        return {"status": "deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#                           ADMIN ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/admin/sessions", dependencies=[Depends(verify_admin)])
async def admin_sessions():
    """Barcha sessiyalar ro'yxati"""
    sm = get_session_manager()
    return await sm.get_all_sessions()


@app.get("/api/v1/admin/resources", dependencies=[Depends(verify_admin)])
async def admin_resources():
    """Tizim resurslari"""
    process = psutil.Process()
    return {
        "memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
        "cpu_percent": process.cpu_percent(),
        "threads": process.num_threads(),
        "uptime_seconds": int(time.time() - _start_time),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                               RUN SERVER
# ═══════════════════════════════════════════════════════════════════════════════

def run_server(port: int = 8090):
    """Server ni ishga tushirish"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=LOG_LEVEL.lower(),
        access_log=False,
    )

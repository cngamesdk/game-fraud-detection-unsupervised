from __future__ import annotations

import hashlib
import json
import pathlib
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

STATIC_DIR = pathlib.Path(__file__).resolve().parent.parent / "static"

from config import settings
from db.connection import close_pool, get_pool
from features.engineering import FeatureEngineer
from model.detector import FraudDetector
from model.storage import ModelStorage
from trace.explainer import RiskExplainer
from listmanager import ListManager


# ---------------------------------------------------------------------------
# Request / response logging middleware with trace ID (contextvars)
# ---------------------------------------------------------------------------

_SKIP_LOG_PREFIXES = ("/static", "/favicon")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if any(request.url.path.startswith(p) for p in _SKIP_LOG_PREFIXES):
            return await call_next(request)

        # Generate trace ID and bind to loguru context — all downstream
        # logger calls within this request will automatically include it.
        trace_id = uuid.uuid4().hex[:12]
        request.state.trace_id = trace_id

        with logger.contextualize(trace_id=trace_id):
            start = time.perf_counter()

            # Read request body for POST (up to 4 KB)
            req_body = ""
            if request.method in ("POST", "PUT"):
                try:
                    raw = await request.body()
                    req_body = raw[:4096].decode("utf-8", errors="replace")
                except Exception:
                    req_body = "<read error>"

            logger.info(
                "--> {method} {path}{body}",
                method=request.method,
                path=request.url.path,
                body=f" | req={req_body}" if req_body else "",
            )

            # Call next and capture response body
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000

            resp_body = b""
            async for chunk in response.body_iterator:
                resp_body += chunk
            resp_text = resp_body[:4096].decode("utf-8", errors="replace")

            logger.info(
                "<-- {status} {ms:.0f}ms | resp={resp}",
                status=response.status_code,
                ms=elapsed_ms,
                resp=resp_text,
            )

            # Inject trace_id into JSON responses
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    data = json.loads(resp_body)
                    if isinstance(data, dict):
                        data["trace_id"] = trace_id
                        resp_body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            return Response(
                content=resp_body,
                status_code=response.status_code,
                headers={k: v for k, v in response.headers.items() if k.lower() != "content-length"},
                media_type=response.media_type,
            )


# ---------------------------------------------------------------------------
# Frontend authentication middleware — cookie-based, password = test source secret
# ---------------------------------------------------------------------------

_FRONTEND_PREFIXES = ("/static",)
_FRONTEND_EXACT = {"/", "/favicon.ico"}
_LOGIN_PATH = "/login"
_AUTH_COOKIE = "fraud_auth_token"
# Use the "test" source secret as the frontend password
_FRONTEND_PASSWORD = settings.SIGN_SECRETS.get("test", "")
# Generate a session token salt at startup
_SESSION_SALT = secrets.token_hex(16)


def _make_auth_token(password: str) -> str:
    """Create a session token from password + salt."""
    return hashlib.sha256(f"{password}:{_SESSION_SALT}".encode()).hexdigest()


class FrontendAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Allow login page and login POST
        if path == _LOGIN_PATH:
            return await call_next(request)

        is_frontend = path in _FRONTEND_EXACT or any(path.startswith(p) for p in _FRONTEND_PREFIXES)
        if not is_frontend:
            return await call_next(request)

        # Check auth cookie
        token = request.cookies.get(_AUTH_COOKIE)
        expected = _make_auth_token(_FRONTEND_PASSWORD)
        if token != expected:
            return RedirectResponse(url=_LOGIN_PATH, status_code=302)

        return await call_next(request)


# ---------------------------------------------------------------------------
# MD5 signature verification middleware — protects prediction endpoints
# ---------------------------------------------------------------------------

class SignatureVerifyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        # Allow requests from logged-in frontend users (valid auth cookie)
        token = request.cookies.get(_AUTH_COOKIE)
        if token and token == _make_auth_token(_FRONTEND_PASSWORD):
            return await call_next(request)

        secrets = settings.SIGN_SECRETS
        if not secrets:
            return await call_next(request)

        source = request.headers.get("X-Source", "")
        timestamp = request.headers.get("X-Timestamp", "")
        sign = request.headers.get("X-Sign", "")

        if not source or not timestamp or not sign:
            return JSONResponse(status_code=401, content={
                "error": "unauthorized", "detail": "Missing X-Source, X-Timestamp, or X-Sign header",
            })

        secret = secrets.get(source)
        if secret is None:
            return JSONResponse(status_code=401, content={
                "error": "unauthorized", "detail": f"Unknown source: {source}",
            })

        try:
            ts = int(timestamp)
        except ValueError:
            return JSONResponse(status_code=401, content={
                "error": "unauthorized", "detail": "Invalid X-Timestamp",
            })
        if abs(time.time() - ts) > settings.SIGN_EXPIRE_SECONDS:
            return JSONResponse(status_code=401, content={
                "error": "unauthorized", "detail": "Signature expired",
            })

        try:
            body = await request.body()
            params = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            params = {}

        sorted_parts = "&".join(f"{k}={params[k]}" for k in sorted(params))
        raw = f"{sorted_parts}&timestamp={timestamp}&secret={secret}"
        expected = hashlib.md5(raw.encode("utf-8")).hexdigest()

        if sign.lower() != expected:
            logger.warning(f"Signature mismatch from source={source}, path={request.url.path}")
            return JSONResponse(status_code=401, content={
                "error": "unauthorized", "detail": "Invalid signature",
            })

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""

    # ── Startup ──────────────────────────────────────────────────────
    logger.info("Starting game fraud detection service...")

    # Initialize MySQL connection pool
    await get_pool()
    logger.info("MySQL connection pool initialized")

    # Model storage
    storage = ModelStorage()

    # Try to load latest model
    detector = storage.load_latest()
    if detector is None:
        logger.warning("No pre-trained model found. Train via POST /api/v1/train")
        detector = FraudDetector()

    feature_engineer = FeatureEngineer()
    explainer = RiskExplainer(detector)

    # Blocklist / Whitelist manager
    listmanager = ListManager()
    try:
        await listmanager.load()
    except Exception:
        logger.warning("Failed to load blocklist from DB — starting with empty sets")

    # Store in app.state for route access
    app.state.detector = detector
    app.state.feature_engineer = feature_engineer
    app.state.explainer = explainer
    app.state.storage = storage
    app.state.listmanager = listmanager

    # Start scheduler
    from scheduler.tasks import start_scheduler
    scheduler = start_scheduler(app)

    logger.info(
        f"Service started | model_fitted={detector.is_fitted} | "
        f"version={detector.version} | port={settings.API_PORT}"
    )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    scheduler.shutdown(wait=False)
    await close_pool()
    logger.info("Service shutdown complete")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="Game Fraud Detection API",
        description="基于无监督学习的游戏风控异常检测服务",
        version="1.0.0",
        lifespan=lifespan,
    )

    from .routes import router
    app.include_router(router)

    # Request/response logging (added first so it wraps everything)
    app.add_middleware(RequestLoggingMiddleware)

    # Security: Frontend cookie-based authentication
    app.add_middleware(FrontendAuthMiddleware)

    # Security: MD5 signature verification for prediction endpoints
    app.add_middleware(SignatureVerifyMiddleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ── Login page ────────────────────────────────────────────────────
    _LOGIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>登录 - 游戏风控系统</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
       min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.login-box { background: #fff; border-radius: 12px; padding: 40px; width: 360px;
             box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
.login-box h2 { text-align: center; margin-bottom: 30px; color: #333; font-size: 22px; }
.form-group { margin-bottom: 20px; }
.form-group label { display: block; margin-bottom: 6px; color: #555; font-size: 14px; }
.form-group input { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px;
                    font-size: 15px; transition: border-color 0.2s; }
.form-group input:focus { outline: none; border-color: #667eea; }
.btn { width: 100%; padding: 12px; background: #667eea; color: #fff; border: none;
       border-radius: 6px; font-size: 16px; cursor: pointer; transition: background 0.2s; }
.btn:hover { background: #5a6fd6; }
.error { color: #e74c3c; font-size: 13px; margin-top: 10px; text-align: center; display: none; }
</style>
</head>
<body>
<div class="login-box">
  <h2>游戏风控异常检测系统</h2>
  <form id="loginForm">
    <div class="form-group">
      <label>密码</label>
      <input type="password" id="password" placeholder="请输入访问密码" autofocus>
    </div>
    <button type="submit" class="btn">登 录</button>
    <p class="error" id="errMsg">密码错误</p>
  </form>
</div>
<script>
document.getElementById('loginForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const pwd = document.getElementById('password').value;
  const resp = await fetch('/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({password: pwd})
  });
  if (resp.ok) {
    window.location.href = '/';
  } else {
    const err = document.getElementById('errMsg');
    err.style.display = 'block';
  }
});
</script>
</body>
</html>"""

    @app.get("/login", include_in_schema=False)
    async def login_page():
        return HTMLResponse(_LOGIN_HTML)

    @app.post("/login", include_in_schema=False)
    async def login_submit(request: Request):
        body = await request.json()
        password = body.get("password", "")
        if password != _FRONTEND_PASSWORD:
            return JSONResponse(status_code=401, content={"error": "密码错误"})
        token = _make_auth_token(_FRONTEND_PASSWORD)
        resp = JSONResponse(content={"ok": True})
        resp.set_cookie(_AUTH_COOKIE, token, httponly=True, max_age=86400 * 7)
        return resp

    @app.get("/logout", include_in_schema=False)
    async def logout():
        resp = RedirectResponse(url=_LOGIN_PATH, status_code=302)
        resp.delete_cookie(_AUTH_COOKIE)
        return resp

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception("Unhandled exception")
        trace_id = getattr(request.state, "trace_id", None)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": str(exc),
                "trace_id": trace_id,
            },
        )

    return app

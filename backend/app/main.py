import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.alerts import router as alerts_router
from app.api.anpr import router as anpr_router
from app.api.auth import router as auth_router
from app.api.cameras import router as cameras_router
from app.api.faces import router as faces_router
from app.api.geofences import router as geofences_router
from app.api.health import router as health_router
from app.api.inference import router as inference_router
from app.api.models import router as models_router
from app.api.stream import router as stream_router
from app.api.tracking import router as tracking_router
from app.core.config import settings
from app.utils.logging import get_logger, setup_logging


setup_logging()
logger = get_logger("SIH26187.API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SIH26187 backend starting.")

    # Initialize alert broadcaster event loop (ContinuousFaceScanner remains OFF by default)
    try:
        from app.pipeline.alert_service import alert_service

        alert_service.broadcaster.set_event_loop(asyncio.get_running_loop())
        logger.info("Alert broadcaster initialized. ContinuousFaceScanner is OFF by default.")
    except Exception as exc:
        logger.warning("Could not initialize alert broadcaster on startup: %s", exc)

    try:
        yield
    finally:
        logger.info("SIH26187 backend shutting down.")
        # Stop background continuous face scanner
        try:
            from app.pipeline.feed_scanner import feed_scanner_service
            feed_scanner_service.stop()
            logger.info("ContinuousFaceScanner stopped.")
        except Exception as exc:
            logger.warning("Could not stop ContinuousFaceScanner on shutdown: %s", exc)

        # Clean up all active ByteTracker sessions
        try:
            from app.tracking.session_store import tracker_session_store
            removed = tracker_session_store.reset_all()
            if removed:
                logger.info("Cleared %d ByteTracker session(s) on shutdown.", removed)
        except Exception as exc:
            logger.warning("Could not clear tracker sessions on shutdown: %s", exc)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-Based Intelligent Video Analytics Backend "
        "for SIH26187 border surveillance."
    ),
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing (CORS) setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.warning(
        "Request validation failed: %s %s | %s",
        request.method,
        request.url.path,
        exc.errors(),
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error": "validation_error",
            "message": "The request contains invalid or missing fields.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled exception: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "internal_server_error",
            "message": "An unexpected internal server error occurred.",
        },
    )


app.include_router(health_router)
app.include_router(auth_router)
app.include_router(cameras_router)
app.include_router(models_router)
app.include_router(inference_router)
app.include_router(stream_router)
app.include_router(tracking_router)
app.include_router(anpr_router)
app.include_router(faces_router)
app.include_router(geofences_router)
app.include_router(alerts_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "status": "online",
        "service": settings.app_name,
        "version": settings.app_version,
    }

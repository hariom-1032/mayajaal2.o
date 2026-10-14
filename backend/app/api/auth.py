"""
FastAPI authentication and session validation router for MayaJaal (MAATRIX).
Enforces operator credential verification, duty shift tokens, and session lifecycles.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.utils.logging import get_logger

logger = get_logger("MAATRIX.Auth")

router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)

# Authorized military & surveillance operator personnel roster
AUTHORIZED_ROSTER = [
    {
        "id": "op-rathore",
        "name": "Admin",
        "rank": "Inspector / General Duty",
        "badgeNumber": "BSF-9482-KR",
        "sector": "Sector-04 (BOP Alpha)",
        "role": "Shift Commander",
        "clearanceLevel": "Level 4 (Top Secret / Restricted)",
        "serviceBranch": "Border Security Force (BSF)",
        "avatarInitials": "AD",
        "passwords": ["MAATRIX2026", "BSF@2026", "COMMANDER", "ADMIN"],
    },
    {
        "id": "op-sharma",
        "name": "Sub-Insp. A. Sharma",
        "rank": "Sub-Inspector / Surveillance",
        "badgeNumber": "ITBP-4108-AS",
        "sector": "Sector-05 (Sikkim Northern LAC)",
        "role": "Tactical Operator",
        "clearanceLevel": "Level 3 (Tactical Secret)",
        "serviceBranch": "Indo-Tibetan Border Police (ITBP)",
        "avatarInitials": "AS",
        "passwords": ["MAATRIX2026", "ITBP@2026", "OPERATOR"],
    },
    {
        "id": "op-verma",
        "name": "Capt. R. Verma",
        "rank": "Operations Officer",
        "badgeNumber": "RAW-7721-RV",
        "sector": "Sector-01 (Punjab Western IB)",
        "role": "Intelligence Analyst",
        "clearanceLevel": "Level 4 (Quantum Grid Crypt)",
        "serviceBranch": "Integrated Border Command (IBC)",
        "avatarInitials": "RV",
        "passwords": ["MAATRIX2026", "RAW@2026", "ANALYST"],
    },
    {
        "id": "op-commander",
        "name": "Brig. V. S. Chauhan",
        "rank": "Sector Commander",
        "badgeNumber": "COMMANDER",
        "sector": "Northern Unified HQ",
        "role": "Shift Commander",
        "clearanceLevel": "Level 5 (Cosmic / Defense Command)",
        "serviceBranch": "Integrated Defense Command (IDC)",
        "avatarInitials": "VC",
        "passwords": ["MAATRIX2026", "COMMANDER@2026", "ADMIN"],
    },
]


class LoginRequest(BaseModel):
    badge: str = Field(..., description="Operator Security Badge ID or Officer Code")
    password: str = Field(..., description="Security Clearance Password")
    remember_me: bool = Field(default=True, description="Extend duty shift duration to 24 hours")


class SessionResponse(BaseModel):
    success: bool
    token: str
    operator: dict[str, Any]
    authenticated_at: str
    expires_at: str


def create_token(operator_dict: dict[str, Any], expires_at: datetime) -> str:
    payload = {
        "id": operator_dict["id"],
        "badge": operator_dict["badgeNumber"],
        "role": operator_dict["role"],
        "name": operator_dict["name"],
        "exp": int(expires_at.timestamp() * 1000),
    }
    dumped = json.dumps(payload)
    return base64.b64encode(dumped.encode("utf-8")).decode("utf-8")


@router.post("/login", response_model=SessionResponse)
async def login(req: LoginRequest) -> SessionResponse:
    """Validate operator credentials and generate an active duty shift session token."""
    clean_badge = req.badge.strip().upper()
    clean_pass = req.password.strip()

    # Look up officer in roster
    matched = None
    for officer in AUTHORIZED_ROSTER:
        if (
            officer["badgeNumber"].upper() == clean_badge
            or officer["id"].upper() == clean_badge
            or (clean_badge == "ADMIN" and (officer["id"] == "op-rathore" or officer["name"] == "Admin" or officer["badgeNumber"] == "COMMANDER"))
        ):
            matched = officer
            break

    if not matched:
        logger.warning("Failed login attempt: Unrecognized badge '%s'", clean_badge)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Badge ID '{clean_badge}' is not recognized in the Border Surveillance Security Registry.",
        )

    valid_password = any(clean_pass.lower() == p.lower() for p in matched["passwords"])
    if not valid_password:
        logger.warning("Failed login attempt: Invalid password for badge '%s'", clean_badge)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: Invalid security clearance password for this badge.",
        )

    now = datetime.now(timezone.utc)
    duration_hours = 24 if req.remember_me else 8
    expires_at = now + timedelta(hours=duration_hours)

    profile = {k: v for k, v in matched.items() if k != "passwords"}
    profile["authenticatedAt"] = now.isoformat()
    profile["expiresAt"] = expires_at.isoformat()

    token = create_token(profile, expires_at)
    profile["token"] = token

    logger.info(
        "Operator '%s' (%s) authenticated successfully for %d-hour shift.",
        profile["name"],
        profile["badgeNumber"],
        duration_hours,
    )

    return SessionResponse(
        success=True,
        token=token,
        operator=profile,
        authenticated_at=now.isoformat(),
        expires_at=expires_at.isoformat(),
    )


@router.get("/session")
async def verify_session(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Verify an active session token and check remaining duty shift time."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
        )

    token = authorization.replace("Bearer ", "").strip()
    try:
        decoded = base64.b64decode(token.encode("utf-8")).decode("utf-8")
        payload = json.loads(decoded)
        exp_ms = payload.get("exp", 0)
        current_ms = int(time.time() * 1000)

        if current_ms >= exp_ms:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Duty shift expired. Please re-authenticate.",
            )

        remaining_sec = max(0, int((exp_ms - current_ms) / 1000))
        return {
            "valid": True,
            "badge": payload.get("badge"),
            "name": payload.get("name"),
            "role": payload.get("role"),
            "remaining_seconds": remaining_sec,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid session token: {exc}",
        )


@router.post("/logout")
async def logout(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Log an operator duty shift termination."""
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        try:
            decoded = base64.b64decode(token.encode("utf-8")).decode("utf-8")
            payload = json.loads(decoded)
            logger.info(
                "Operator '%s' (%s) ended duty shift / logged out.",
                payload.get("name"),
                payload.get("badge"),
            )
        except Exception:
            pass

    return {"success": True, "message": "Duty shift session terminated successfully."}


@router.get("/roster")
async def get_roster() -> list[dict[str, Any]]:
    """Return public roster metadata for authorized demo personnel."""
    return [
        {k: v for k, v in officer.items() if k != "passwords"}
        for officer in AUTHORIZED_ROSTER
    ]

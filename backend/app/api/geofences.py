"""
Geofence & Directional Tripwire REST API Router.

Provides CRUD management for tactical security zones and tripwires,
as well as a simulation / evaluation endpoint for interactive UI testing.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.pipeline.geofence_engine import geofence_engine

logger = logging.getLogger("SIH26187.GeofencesAPI")

router = APIRouter(prefix="/api/geofences", tags=["Geofences"])


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ZoneCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, description="Designation name of the perimeter zone")
    cameraId: str = Field("bop-jk-01", description="Camera ID this zone applies to")
    color: str = Field("#ef4444", description="Hex color code for UI rendering")
    severity: str = Field("CRITICAL", description="CRITICAL, HIGH, MEDIUM, or LOW")
    polygon: list[list[float]] = Field(..., min_items=3, description="List of [x, y] normalized vertices")
    targetClasses: list[str] = Field(default=["person", "car", "truck", "suspect"])
    accessMode: Optional[str] = Field("AUTHORIZED_PASS", description="AUTHORIZED_PASS, ALL_ALERTS, or SUSPECT_ONLY")
    enabled: bool = Field(True)
    cooldownSeconds: float = Field(20.0, ge=1.0, le=300.0)
    description: Optional[str] = None


class ZoneUpdateRequest(BaseModel):
    name: Optional[str] = None
    cameraId: Optional[str] = None
    color: Optional[str] = None
    severity: Optional[str] = None
    polygon: Optional[list[list[float]]] = None
    targetClasses: Optional[list[str]] = None
    accessMode: Optional[str] = None
    enabled: Optional[bool] = None
    cooldownSeconds: Optional[float] = None
    description: Optional[str] = None


class TripwireCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, description="Designation name of the tripwire")
    cameraId: str = Field("bop-jk-01", description="Camera ID this tripwire applies to")
    color: str = Field("#dc2626", description="Hex color code for UI rendering")
    severity: str = Field("CRITICAL", description="CRITICAL, HIGH, MEDIUM, or LOW")
    p1: list[float] = Field(..., min_items=2, max_items=2, description="Start point [x, y] normalized")
    p2: list[float] = Field(..., min_items=2, max_items=2, description="End point [x, y] normalized")
    direction: str = Field("FORWARD", description="FORWARD, REVERSE, or BIDIRECTIONAL")
    targetClasses: list[str] = Field(default=["person", "suspect"])
    accessMode: Optional[str] = Field("AUTHORIZED_PASS", description="AUTHORIZED_PASS, ALL_ALERTS, or SUSPECT_ONLY")
    enabled: bool = Field(True)
    cooldownSeconds: float = Field(15.0, ge=1.0, le=300.0)
    description: Optional[str] = None


class TripwireUpdateRequest(BaseModel):
    name: Optional[str] = None
    cameraId: Optional[str] = None
    color: Optional[str] = None
    severity: Optional[str] = None
    p1: Optional[list[float]] = None
    p2: Optional[list[float]] = None
    direction: Optional[str] = None
    targetClasses: Optional[list[str]] = None
    accessMode: Optional[str] = None
    enabled: Optional[bool] = None
    cooldownSeconds: Optional[float] = None
    description: Optional[str] = None


class EvaluateTrackItem(BaseModel):
    trackId: int = Field(..., description="Unique track ID")
    box: list[float] = Field(..., min_items=4, description="Bounding box [x1, y1, x2, y2]")
    className: str = Field("person")
    confidence: float = Field(0.95, ge=0.0, le=1.0)
    suspectName: Optional[str] = None
    personName: Optional[str] = None
    identityStatus: Optional[str] = None
    isAuthorized: Optional[bool] = None
    isSuspect: Optional[bool] = None
    isUnknown: Optional[bool] = None


class EvaluateRequest(BaseModel):
    cameraId: str = Field(..., description="Camera ID to evaluate against")
    tracks: list[EvaluateTrackItem] = Field(..., description="Active tracks to evaluate")
    dispatchAlerts: bool = Field(False, description="Whether to publish real alerts if breach detected")


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("")
async def get_geofences(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID")
) -> dict[str, Any]:
    """Retrieve all virtual geofences and tripwires with optional camera filtering."""
    return geofence_engine.get_all(camera_id=camera_id)


# ── Zones ────────────────────────────────────────────────────────────────────

@router.post("/zones", status_code=status.HTTP_201_CREATED)
async def create_zone(payload: ZoneCreateRequest) -> dict[str, Any]:
    """Create a new polygonal virtual geofence."""
    created = geofence_engine.add_zone(payload.dict())
    return {"status": "success", "zone": created}


@router.put("/zones/{zone_id}")
async def update_zone(zone_id: str, payload: ZoneUpdateRequest) -> dict[str, Any]:
    """Update an existing virtual geofence polygon by ID."""
    updates = {k: v for k, v in payload.dict().items() if v is not None}
    updated = geofence_engine.update_zone(zone_id, updates)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Geofence zone '{zone_id}' not found.",
        )
    return {"status": "success", "zone": updated}


@router.delete("/zones/{zone_id}")
async def delete_zone(zone_id: str) -> dict[str, Any]:
    """Delete a virtual geofence polygon by ID."""
    success = geofence_engine.delete_zone(zone_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Geofence zone '{zone_id}' not found.",
        )
    return {"status": "success", "deleted_id": zone_id}


# ── Tripwires ────────────────────────────────────────────────────────────────

@router.post("/tripwires", status_code=status.HTTP_201_CREATED)
async def create_tripwire(payload: TripwireCreateRequest) -> dict[str, Any]:
    """Create a new directional tripwire."""
    created = geofence_engine.add_tripwire(payload.dict())
    return {"status": "success", "tripwire": created}


@router.put("/tripwires/{wire_id}")
async def update_tripwire(wire_id: str, payload: TripwireUpdateRequest) -> dict[str, Any]:
    """Update an existing directional tripwire by ID."""
    updates = {k: v for k, v in payload.dict().items() if v is not None}
    updated = geofence_engine.update_tripwire(wire_id, updates)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tripwire '{wire_id}' not found.",
        )
    return {"status": "success", "tripwire": updated}


@router.delete("/tripwires/{wire_id}")
async def delete_tripwire(wire_id: str) -> dict[str, Any]:
    """Delete a directional tripwire by ID."""
    success = geofence_engine.delete_tripwire(wire_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tripwire '{wire_id}' not found.",
        )
    return {"status": "success", "deleted_id": wire_id}


# ── Interactive Simulation / Evaluation ──────────────────────────────────────

@router.post("/evaluate")
async def evaluate_geofences(payload: EvaluateRequest) -> dict[str, Any]:
    """
    Evaluate tracks against configured geofences and tripwires for a camera.
    Ideal for real-time video pipeline hooks and frontend breach simulation.
    """
    tracked_objects = [
        {
            "track_id": t.trackId,
            "box": t.box,
            "class_name": t.className,
            "confidence": t.confidence,
            "suspect_name": t.suspectName,
            "person_name": t.personName,
            "identity_status": t.identityStatus,
            "is_authorized": t.isAuthorized,
            "is_suspect": t.isSuspect,
            "is_unknown": t.isUnknown,
        }
        for t in payload.tracks
    ]

    breaches = geofence_engine.evaluate_tracks(
        camera_id=payload.cameraId,
        tracked_objects=tracked_objects,
        auto_alert=payload.dispatchAlerts,
    )

    return {
        "status": "success",
        "cameraId": payload.cameraId,
        "evaluatedTracks": len(tracked_objects),
        "breachesCount": len(breaches),
        "breaches": breaches,
    }


@router.post("/reset-history")
async def reset_geofence_history(
    camera_id: Optional[str] = Query(None, description="Specific camera ID to reset")
) -> dict[str, Any]:
    """Clear in-memory trajectory history and alert cooldowns."""
    geofence_engine.reset_history(camera_id=camera_id)
    return {"status": "success", "message": "Geofence history and cooldowns reset."}

"use client";

import { useEffect, useRef } from "react";
import { Detection } from "@/types/backend";
import { formatConfidence } from "@/lib/utils";

interface DetectionCanvasProps {
  detections: Detection[];
  sourceWidth: number;
  sourceHeight: number;
  className?: string;
}

const CLASS_COLORS: Record<string, string> = {
  // Primary Border Threat Classes
  firearm: "#ef4444", // High threat - Red
  explosive: "#dc2626", // Extreme threat - Dark Crimson
  melee_weapon: "#f43f5e", // Threat - Rose
  blunt_weapon: "#a855f7", // Threat - Purple
  fire_smoke: "#f97316", // Hazard - Orange
  person: "#10b981", // Target - Emerald green
  tool: "#06b6d4", // Object - Cyan

  // Standard COCO fallback categories
  car: "#38bdf8", // Sky blue
  truck: "#0284c7", // Deep blue
  bus: "#6366f1", // Indigo
  motorcycle: "#06b6d4", // Cyan
  bicycle: "#14b8a6", // Teal
  dog: "#f59e0b", // Amber
  horse: "#d97706", // Dark amber
  backpack: "#ec4899", // Pink
  suitcase: "#a855f7", // Purple
  default: "#22c55e", // Light green
};

export function DetectionCanvas({
  detections,
  sourceWidth,
  sourceHeight,
  className = "",
}: DetectionCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, rect.width, rect.height);

      if (!detections || detections.length === 0 || sourceWidth <= 0 || sourceHeight <= 0) {
        ctx.restore();
        return;
      }

      // Handle CSS object-contain letterboxing / pillarboxing
      const canvasWidth = rect.width;
      const canvasHeight = rect.height;
      const containerAspect = canvasWidth / canvasHeight;
      const imageAspect = sourceWidth / sourceHeight;

      let renderWidth = canvasWidth;
      let renderHeight = canvasHeight;
      let offsetX = 0;
      let offsetY = 0;

      if (imageAspect > containerAspect) {
        // Image is wider than container: letterboxed on top and bottom
        renderWidth = canvasWidth;
        renderHeight = canvasWidth / imageAspect;
        offsetX = 0;
        offsetY = (canvasHeight - renderHeight) / 2;
      } else {
        // Image is taller than container: pillarboxed on left and right
        renderHeight = canvasHeight;
        renderWidth = canvasHeight * imageAspect;
        offsetX = (canvasWidth - renderWidth) / 2;
        offsetY = 0;
      }

      const scaleX = renderWidth / sourceWidth;
      const scaleY = renderHeight / sourceHeight;

      // Enforce physical invariant: deduplicate multiple detections claiming the same known identity in the same frame
      const claimedIdentities = new Set<string>();
      const processedDetections = detections.map((det) => {
        const rawName = (det.suspect_name || det.class_name.replace(/^(face_|suspect_)/i, "")).trim().toLowerCase();
        const isIdentityTagged = det.class_name.toLowerCase().startsWith("face_") || det.class_name.toLowerCase().startsWith("suspect_");
        if (isIdentityTagged && rawName && rawName !== "unknown") {
          if (claimedIdentities.has(rawName)) {
            // Identity collision in same frame: demote duplicate to Unknown
            return {
              ...det,
              class_name: "face_unknown",
              suspect_name: "Unknown",
              is_threat: false,
              threat_level: undefined,
            };
          }
          claimedIdentities.add(rawName);
        }
        return det;
      });

      processedDetections.forEach((det) => {
        const [x1, y1, x2, y2] = det.box;
        const rx1 = offsetX + x1 * scaleX;
        const ry1 = offsetY + y1 * scaleY;
        const rx2 = offsetX + x2 * scaleX;
        const ry2 = offsetY + y2 * scaleY;
        const boxW = rx2 - rx1;
        const boxH = ry2 - ry1;

        const isSuspect = Boolean(det.is_threat || det.class_name.toLowerCase().startsWith("suspect") || det.threat_level);
        const isUnknownFace = det.class_name.toLowerCase().includes("unknown") || det.suspect_name?.toLowerCase() === "unknown";
        const isFace = (det.class_name.toLowerCase().startsWith("face_") || det.class_name.toLowerCase().startsWith("person_")) && !isUnknownFace;

        let color = CLASS_COLORS[det.class_name.toLowerCase()] || CLASS_COLORS.default;
        if (isSuspect) {
          color = "#dc2626"; // Crimson Alert
        } else if (isFace) {
          color = "#10b981"; // Emerald Known Face
        } else if (isUnknownFace) {
          color = "#94a3b8"; // Neutral Gray Unknown Face
        }

        // 1. Semi-transparent fill
        ctx.fillStyle = isSuspect ? `${color}28` : `${color}18`;
        ctx.fillRect(rx1, ry1, boxW, boxH);

        // 2. Bounding Box Stroke
        ctx.lineWidth = isSuspect ? 3 : 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(rx1, ry1, boxW, boxH);

        // 3. Tactical Corner brackets
        const cornerLen = Math.min(14, Math.max(4, boxW / 4), Math.max(4, boxH / 4));
        ctx.lineWidth = 3;
        ctx.strokeStyle = isSuspect ? "#fca5a5" : "#ffffff";

        // Top-Left corner
        ctx.beginPath();
        ctx.moveTo(rx1, ry1 + cornerLen);
        ctx.lineTo(rx1, ry1);
        ctx.lineTo(rx1 + cornerLen, ry1);
        ctx.stroke();

        // Top-Right corner
        ctx.beginPath();
        ctx.moveTo(rx2 - cornerLen, ry1);
        ctx.lineTo(rx2, ry1);
        ctx.lineTo(rx2, ry1 + cornerLen);
        ctx.stroke();

        // Bottom-Left corner
        ctx.beginPath();
        ctx.moveTo(rx1, ry2 - cornerLen);
        ctx.lineTo(rx1, ry2);
        ctx.lineTo(rx1 + cornerLen, ry2);
        ctx.stroke();

        // Bottom-Right corner
        ctx.beginPath();
        ctx.moveTo(rx2 - cornerLen, ry2);
        ctx.lineTo(rx2, ry2);
        ctx.lineTo(rx2, ry2 - cornerLen);
        ctx.stroke();

        // 4. Label Badge
        const confStr = formatConfidence(det.confidence);
        const trackTag = det.track_id !== undefined ? `ID #${det.track_id} · ` : "";
        let labelText = `${trackTag}${det.class_name.replace(/_/g, " ").toUpperCase()} ${confStr}`;

        if (isSuspect) {
          const name = (det.suspect_name || det.class_name.replace(/^suspect_/i, "")).toUpperCase();
          const tLevel = det.threat_level || "ALERT";
          labelText = `🚨 SUSPECT: ${name} [${tLevel}] ${confStr}`;
        } else if (isFace) {
          const name = (det.suspect_name || det.class_name.replace(/^face_/i, "")).toUpperCase();
          labelText = `👤 ${name} ${confStr}`;
        } else if (isUnknownFace) {
          labelText = `👤 UNKNOWN ${confStr}`;
        }

        ctx.font = isSuspect ? "bold 11px system-ui, sans-serif" : "bold 11px monospace";
        const textMetrics = ctx.measureText(labelText);
        const paddingX = 6;
        const badgeH = 20;
        const badgeW = textMetrics.width + paddingX * 2;
        const badgeY = ry1 - badgeH >= offsetY ? ry1 - badgeH : ry1;

        ctx.fillStyle = color;
        ctx.fillRect(rx1, badgeY, badgeW, badgeH);

        if (det.track_id !== undefined || isSuspect) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1;
          ctx.strokeRect(rx1, badgeY, badgeW, badgeH);
        }

        ctx.fillStyle = "#ffffff";
        ctx.textBaseline = "middle";
        ctx.fillText(labelText, rx1 + paddingX, badgeY + badgeH / 2);
      });

      ctx.restore();
    };

    draw();

    const resizeObserver = new ResizeObserver(() => {
      draw();
    });
    resizeObserver.observe(canvas);

    return () => {
      resizeObserver.disconnect();
    };
  }, [detections, sourceWidth, sourceHeight]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none w-full h-full z-20 ${className}`}
    />
  );
}

export default DetectionCanvas;

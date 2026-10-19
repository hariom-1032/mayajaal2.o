"use client";

import { CameraCard } from "./cameracard";
import { useCameras } from "@/lib/camerasStore";
import { useAlerts } from "@/lib/alertsStore";
import { Camera } from "@/types/camera";

interface CameraGridProps {
  selectedCameraId?: string;
  onSelectCamera?: (camera: Camera) => void;
}

export function CameraGrid({ selectedCameraId, onSelectCamera }: CameraGridProps) {
  const { cameras } = useCameras();
  const { alerts } = useAlerts();

  // Index active, unacknowledged suspect alerts by camera ID
  const suspectAlertMap = new Map<string, typeof alerts[0]>();
  alerts.forEach((alert) => {
    if (!alert.acknowledged && (alert.suspectName || alert.className === "suspect") && alert.cameraId) {
      if (!suspectAlertMap.has(alert.cameraId)) {
        suspectAlertMap.set(alert.cameraId, alert);
      }
    }
  });

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4" suppressHydrationWarning>
      {cameras.map((camera) => {
        const suspectHit = suspectAlertMap.get(camera.id);

        return (
          <CameraCard
            key={camera.id}
            id={camera.id}
            name={camera.name}
            location={camera.location}
            sector={camera.sector}
            isOnline={camera.status !== "offline"}
            streamUrl={camera.streamUrl}
            ipAddress={camera.ipAddress}
            isSelected={camera.id === selectedCameraId}
            hasSuspectAlert={Boolean(suspectHit)}
            suspectName={suspectHit?.suspectName}
            threatLevel={suspectHit?.threatLevel}
            onClick={() => onSelectCamera?.(camera)}
          />
        );
      })}
    </div>
  );
}

export default CameraGrid;


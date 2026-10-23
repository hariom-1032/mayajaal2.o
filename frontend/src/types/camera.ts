export type CameraStatus = "online" | "offline" | "degraded" | "alert";

export type CameraType =
  | "Optical 4K"
  | "Thermal FLIR"
  | "Night Vision / IR"
  | "ANPR Dedicated"
  | "PTZ 360"
  | "Panoramic";

/**
 * Core Camera entity — only 5 user-facing fields:
 *   1. name       – human label (e.g. "CSD", "MAIN OFFICE OUTDOOR")
 *   2. type       – sensor category
 *   3. status     – online | offline | degraded | alert
 *   4. channel    – NVR channel number (drives RTSP URL)
 *   5. latitude / longitude – nullable GPS coordinates
 *
 * Plus internal plumbing: id, streamUrl.
 */
export interface Camera {
  id: string;
  name: string;
  channel?: number;
  type?: CameraType | string;
  status: CameraStatus;
  latitude: number | null;
  longitude: number | null;
  streamUrl?: string;
  /** @deprecated kept only for backend compat — not displayed in UI */
  sector?: string;
  /** @deprecated kept only for backend compat — not displayed in UI */
  location?: string;
  /** @deprecated kept only for backend compat */
  coordinates?: [number, number];
  /** @deprecated kept only for backend compat */
  ipAddress?: string;
  /** @deprecated kept only for backend compat */
  port?: number;
  /** @deprecated kept only for backend compat */
  isRtsp?: boolean;
  /** @deprecated kept only for backend compat */
  modelAssigned?: string;
  /** @deprecated kept only for backend compat */
  resolution?: string;
  /** @deprecated kept only for backend compat */
  fps?: number;
  /** @deprecated kept only for backend compat */
  confThreshold?: number;
  /** @deprecated kept only for backend compat */
  iouThreshold?: number;
  /** @deprecated kept only for backend compat */
  isRecording?: boolean;
  /** @deprecated kept only for backend compat */
  alertTriggerEnabled?: boolean;
  /** @deprecated kept only for backend compat */
  lastActive?: string;
  /** @deprecated kept only for backend compat */
  healthStats?: { bitrate: string; latencyMs: number; packetLoss: string };
}

export interface CreateCameraInput {
  id?: string;
  name: string;
  channel?: number;
  type?: CameraType | string;
  status?: CameraStatus;
  latitude?: number | null;
  longitude?: number | null;
  streamUrl?: string;

  /** @deprecated legacy fields for backwards compatibility */
  sector?: string;
  location?: string;
  ipAddress?: string;
  port?: number;
  modelAssigned?: string;
  resolution?: string;
  fps?: number;
  confThreshold?: number;
  iouThreshold?: number;
}

export interface UpdateCameraInput {
  name?: string;
  channel?: number;
  type?: CameraType | string;
  status?: CameraStatus;
  latitude?: number | null;
  longitude?: number | null;
  streamUrl?: string;

  /** @deprecated legacy fields for backwards compatibility */
  sector?: string;
  location?: string;
  ipAddress?: string;
  port?: number;
  modelAssigned?: string;
  resolution?: string;
  fps?: number;
  confThreshold?: number;
  iouThreshold?: number;
}

export interface StreamTestResult {
  reachable: boolean;
  status: "online" | "unreachable" | "error";
  latencyMs: number;
  host?: string;
  port?: number;
  message: string;
  protocol?: string;
  supportedCodecs?: string[];
}

export type AlertSeverity = "High" | "Medium" | "Low";

export interface QrtDispatchRecord {
  unit: string;
  notes?: string;
  dispatched_at: number;
  status: "Dispatched" | "Arrived" | "Resolved";
}

export interface AlertItem {
  id: string;
  title: string;
  location: string;
  time: string;
  timestamp: number;
  severity: AlertSeverity;
  cameraId?: string;
  cameraName?: string;
  className?: string;
  confidence?: number;
  acknowledged?: boolean;
  box?: [number, number, number, number] | number[];
  snapshotUrl?: string;
  suspectName?: string;
  threatLevel?: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | string;
  category?: string;
  notes?: string;
  qrt_dispatch?: QrtDispatchRecord;
}

export interface AlertFilter {
  severity?: AlertSeverity | "All";
  search?: string;
  acknowledged?: boolean;
}

export interface ScannerStatus {
  running: boolean;
  worker_count: number;
  active_camera_ids: string[];
  total_scans: number;
  suspect_detections: number;
  last_scan_time: string;
  sampling_interval_sec: number;
  debounce_cooldown_sec: number;
}

export interface SuspectWaypoint {
  step: number;
  alert_id?: string;
  camera_id: string;
  camera_name: string;
  location?: string;
  sector?: string;
  latitude: number;
  longitude: number;
  timestamp: number;
  time_str: string;
  threat_level?: string;
  confidence: number;
  snapshot_url?: string;
  delta_km?: number;
  delta_distance_km?: number;
  elapsed_minutes?: number;
  elapsed_time_minutes?: number;
  reid_score?: number;
  body_score?: number;
  face_score?: number;
  fusion_mode?: string;
  has_face?: boolean;
}

export interface SuspectTrajectory {
  suspect_name?: string;
  trace_id?: string;
  found: boolean;
  threat_level?: string;
  category?: string;
  total_sightings: number;
  total_distance_km: number;
  first_seen?: string;
  last_seen?: string;
  last_location?: string;
  last_seen_time?: string;
  waypoints: SuspectWaypoint[];
  is_cross_camera?: boolean;
  has_face_fused?: boolean;
  fusion_mode?: string;
  demographics?: {
    gender?: string;
    age_group?: string;
    race?: string;
  };
  attributes?: {
    has_disguise?: boolean;
    disguise_labels?: string[];
    mask_prob?: number;
    sunglasses_prob?: number;
  };
  liveness?: {
    is_real?: boolean;
    confidence?: number;
  };
}

export interface GlobalTraceItem {
  trace_id: string;
  first_seen: number;
  last_seen: number;
  duration_sec: number;
  last_camera_id: string;
  last_camera_name: string;
  total_sightings: number;
  unique_cameras_count: number;
  is_cross_camera: boolean;
  is_suspect: boolean;
  suspect_name?: string | null;
  threat_level?: string | null;
  category?: string | null;
  waypoints_count: number;
  has_face_fused?: boolean;
  fusion_mode?: string;
  last_body_score?: number;
  last_face_score?: number;
  last_fused_score?: number;
  demographics?: {
    gender?: string;
    age_group?: string;
    race?: string;
  };
  attributes?: {
    has_disguise?: boolean;
    disguise_labels?: string[];
  };
  liveness?: {
    is_real?: boolean;
    confidence?: number;
  };
}




export type ThreatLevel = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

export interface EnrolledPerson {
  id: string;
  name: string;
  image_url: string;
  created_at: string;
  sample_count: number;
  is_suspect?: boolean;
  threat_level?: ThreatLevel;
  category?: string;
  notes?: string;
}

export interface FaceDetection {
  name: string;
  is_known: boolean;
  confidence: number;
  match_score: number;
  l2_distance?: number;
  calibrated_conf: number;
  bbox: [number, number, number, number];
  type: string;
  class_name: string;
  is_threat: boolean;
  threat_level?: string;
  category?: string;
}

export interface FaceScanResponse {
  success: boolean;
  face_count: number;
  faces: FaceDetection[];
  latency_ms: number;
  annotated_image?: string;
  error?: string;
}

export interface FaceEngineStatus {
  status: "ready" | "degraded";
  detector_loaded: boolean;
  recognizer_loaded: boolean;
  enrolled_count: number;
  yunet_model: string;
  sface_model: string;
}

export interface FaceEvent {
  id: string;
  name: string;
  is_known: boolean;
  calibrated_conf: number;
  timestamp: string;
  bbox: [number, number, number, number];
  is_threat?: boolean;
  threat_level?: string;
  category?: string;
}

export interface RegisterFaceResponse {
  success: boolean;
  person?: EnrolledPerson;
  message?: string;
  error?: string;
}

export interface UnknownPerson {
  id: string;
  timestamp: string;
  iso_timestamp: string;
  camera_id: string;
  camera_name: string;
  cameras_seen?: string[];
  confidence: number;
  bbox: [number, number, number, number];
  photo_filename: string;
  photo_url: string;
  seen_count: number;
  last_seen: string;
  status: "unreviewed" | "flagged" | "enrolled" | "dismissed";
  notes?: string;
  enrolled_name?: string | null;
}

export interface UnknownStats {
  success: boolean;
  total_unknowns: number;
  today_count: number;
  unreviewed_count: number;
  enrolled_count: number;
  by_camera: Record<string, number>;
}


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

export interface FaceDemographics {
  gender?: string;
  age_group?: string;
  race?: string;
}

export interface FaceAttributes {
  mask_prob?: number;
  sunglasses_prob?: number;
  eyeglasses_prob?: number;
  has_disguise?: boolean;
  disguise_labels?: string[];
}

export interface FaceLiveness {
  is_real: boolean;
  confidence: number;
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
  engine?: string;
  demographics?: FaceDemographics;
  attributes?: FaceAttributes;
  liveness?: FaceLiveness;
}

export interface FaceScanResponse {
  success: boolean;
  engine?: string;
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
  uniface_ready?: boolean;
  active_engine?: string;
  enrolled_count: number;
  yunet_model?: string;
  sface_model?: string;
  uniface_models?: Record<string, string>;
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
  demographics?: FaceDemographics;
  attributes?: FaceAttributes;
  liveness?: FaceLiveness;
  engine?: string;
}

export interface RegisterFaceResponse {
  success: boolean;
  person?: EnrolledPerson;
  message?: string;
  error?: string;
}


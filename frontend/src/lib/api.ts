import {
  HealthResponse,
  InferenceResponse,
  InferenceStatusResponse,
  ModelStatusResponse,
  RTSPInferenceResponse,
  RTSPTrackingResponse,
  TrackingVideoResponse,
  VideoInferenceResponse,
} from "@/types/backend";
import {
  Camera,
  CreateCameraInput,
  UpdateCameraInput,
  StreamTestResult,
} from "@/types/camera";
import {
  AnprRecord,
  AnprScanResponse,
  AnprVideoResponse,
  WatchlistEntry,
} from "@/types/anpr";
import {
  EnrolledPerson,
  FaceEngineStatus,
  FaceEvent,
  FaceScanResponse,
  RegisterFaceResponse,
} from "@/types/face";
import { AlertItem, ScannerStatus, QrtDispatchRecord, SuspectTrajectory, GlobalTraceItem } from "@/types/alert";
import {
  GeofenceZone,
  DirectionalTripwire,
  GeofencesResponse,
  GeofenceEvaluateResponse,
  CreateZonePayload,
  CreateTripwirePayload,
} from "@/types/geofence";

const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${BACKEND_BASE_URL}${endpoint}`;
  try {
    const timeoutSignal =
      typeof AbortSignal !== "undefined" && "timeout" in AbortSignal
        ? AbortSignal.timeout(6000)
        : undefined;

    const res = await fetch(url, {
      ...options,
      signal: options.signal || timeoutSignal,
      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
    });

    if (!res.ok) {
      let errorDetail = "";
      try {
        const errorJson = await res.json();
        errorDetail =
          errorJson.message || errorJson.detail || JSON.stringify(errorJson);
      } catch {
        errorDetail = await res.text();
      }
      throw new ApiError(
        `API ${options.method || "GET"} ${endpoint} failed (${res.status}): ${errorDetail}`,
        res.status,
        errorDetail
      );
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    const message = err instanceof Error ? err.message : String(err);
    throw new ApiError(
      `Network error contacting backend at ${url}: ${message}`
    );
  }
}

export interface StreamValidationResult {
  reachable: boolean;
  status: "ready" | "online" | "unreachable";
  protocol: string;
  target_url: string;
  resolved_url: string;
  host?: string;
  port?: number;
  latency_ms: number;
  message: string;
  is_ip_webcam: boolean;
  candidates?: string[];
}

export const api = {
  /**
   * Health and Provider status
   */
  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/api/health");
  },

  /**
   * Model Manager status and loaded models
   */
  async getModels(): Promise<ModelStatusResponse> {
    return request<ModelStatusResponse>("/api/models");
  },

  /**
   * Load an ONNX model file located in backend models directory
   */
  async loadModel(
    modelName: string,
    modelFile: string
  ): Promise<{ model_name: string; status: string; model_info: unknown }> {
    return request("/api/models/load", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model_name: modelName,
        model_file: modelFile,
      }),
    });
  },

  /**
   * Unload a model by name
   */
  async unloadModel(
    modelName: string
  ): Promise<{ model_name: string; status: string }> {
    return request(`/api/models/${encodeURIComponent(modelName)}`, {
      method: "DELETE",
    });
  },

  /**
   * Inference and resource slot status
   */
  async getInferenceStatus(): Promise<InferenceStatusResponse> {
    return request<InferenceStatusResponse>("/api/inference/status");
  },

  /**
   * Run object detection inference on a single image file
   */
  async runImageInference(params: {
    file: File | Blob;
    modelName: string;
    confThreshold?: number;
    iouThreshold?: number;
    postprocess?: boolean;
  }): Promise<InferenceResponse> {
    const formData = new FormData();
    formData.append("file", params.file);
    formData.append("model_name", params.modelName);
    formData.append(
      "conf_threshold",
      (params.confThreshold ?? 0.25).toString()
    );
    formData.append(
      "iou_threshold",
      (params.iouThreshold ?? 0.45).toString()
    );
    formData.append(
      "postprocess",
      (params.postprocess ?? true).toString()
    );

    return request<InferenceResponse>("/api/inference/image", {
      method: "POST",
      body: formData,
    });
  },

  /**
   * Run inference on an uploaded video file
   */
  async runVideoInference(params: {
    file: File | Blob;
    modelName: string;
    confThreshold?: number;
    iouThreshold?: number;
    frameStride?: number;
    maxFrames?: number;
    postprocess?: boolean;
  }): Promise<VideoInferenceResponse> {
    const formData = new FormData();
    formData.append("file", params.file);
    formData.append("model_name", params.modelName);
    if (params.confThreshold !== undefined)
      formData.append("conf_threshold", params.confThreshold.toString());
    if (params.iouThreshold !== undefined)
      formData.append("iou_threshold", params.iouThreshold.toString());
    if (params.frameStride !== undefined)
      formData.append("frame_stride", params.frameStride.toString());
    if (params.maxFrames !== undefined)
      formData.append("max_frames", params.maxFrames.toString());
    if (params.postprocess !== undefined)
      formData.append("postprocess", params.postprocess.toString());

    return request<VideoInferenceResponse>("/api/inference/video", {
      method: "POST",
      body: formData,
    });
  },

  /**
   * Run sampled inference on an RTSP stream URL
   */
  async runRtspInference(params: {
    rtspUrl: string;
    modelName: string;
    confThreshold?: number;
    iouThreshold?: number;
    maxFrames?: number;
  }): Promise<RTSPInferenceResponse> {
    const formData = new FormData();
    formData.append("rtsp_url", params.rtspUrl);
    formData.append("model_name", params.modelName);
    if (params.confThreshold !== undefined)
      formData.append("conf_threshold", params.confThreshold.toString());
    if (params.iouThreshold !== undefined)
      formData.append("iou_threshold", params.iouThreshold.toString());
    if (params.maxFrames !== undefined)
      formData.append("max_frames", params.maxFrames.toString());

    return request<RTSPInferenceResponse>("/api/inference/rtsp", {
      method: "POST",
      body: formData,
    });
  },

  /**
   * Run ByteTrack multi-object tracking on an uploaded video file
   */
  async runVideoTracking(params: {
    file: File | Blob;
    modelName: string;
    confThreshold?: number;
    iouThreshold?: number;
    maxFrames?: number;
    activationThreshold?: number;
    lostTrackBuffer?: number;
    matchingThreshold?: number;
  }): Promise<TrackingVideoResponse> {
    const formData = new FormData();
    formData.append("file", params.file);
    formData.append("model_name", params.modelName);
    if (params.confThreshold !== undefined)
      formData.append("conf_threshold", params.confThreshold.toString());
    if (params.iouThreshold !== undefined)
      formData.append("iou_threshold", params.iouThreshold.toString());
    if (params.maxFrames !== undefined)
      formData.append("max_frames", params.maxFrames.toString());
    if (params.activationThreshold !== undefined)
      formData.append("activation_threshold", params.activationThreshold.toString());
    if (params.lostTrackBuffer !== undefined)
      formData.append("lost_track_buffer", params.lostTrackBuffer.toString());
    if (params.matchingThreshold !== undefined)
      formData.append("matching_threshold", params.matchingThreshold.toString());

    return request<TrackingVideoResponse>("/api/tracking/video", {
      method: "POST",
      body: formData,
    });
  },

  /**
   * Run ByteTrack tracking on an RTSP stream with persistent camera tracking state
   */
  async runRtspTracking(params: {
    rtspUrl: string;
    cameraId?: string;
    modelName: string;
    confThreshold?: number;
    iouThreshold?: number;
    maxFrames?: number;
  }): Promise<RTSPTrackingResponse> {
    const formData = new FormData();
    formData.append("rtsp_url", params.rtspUrl);
    formData.append("model_name", params.modelName);
    if (params.cameraId) formData.append("camera_id", params.cameraId);
    if (params.confThreshold !== undefined)
      formData.append("conf_threshold", params.confThreshold.toString());
    if (params.iouThreshold !== undefined)
      formData.append("iou_threshold", params.iouThreshold.toString());
    if (params.maxFrames !== undefined)
      formData.append("max_frames", params.maxFrames.toString());

    return request<RTSPTrackingResponse>("/api/tracking/rtsp", {
      method: "POST",
      body: formData,
    });
  },

  /**
   * Cameras Management API
   */
  async getCameras(filters?: { sector?: string; status?: string }): Promise<Camera[]> {
    const params = new URLSearchParams();
    if (filters?.sector) params.append("sector", filters.sector);
    if (filters?.status) params.append("status", filters.status);
    const qs = params.toString() ? `?${params.toString()}` : "";
    return request<Camera[]>(`/api/cameras${qs}`);
  },

  async getCamera(id: string): Promise<Camera> {
    return request<Camera>(`/api/cameras/${encodeURIComponent(id)}`);
  },

  async createCamera(input: CreateCameraInput): Promise<Camera> {
    return request<Camera>("/api/cameras", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
  },

  async updateCamera(id: string, input: UpdateCameraInput): Promise<Camera> {
    return request<Camera>(`/api/cameras/${encodeURIComponent(id)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
  },

  async deleteCamera(id: string): Promise<{ status: string; message: string; deleted_id: string }> {
    return request(`/api/cameras/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
  },

  async testStream(params: { streamUrl?: string; ipAddress?: string; port?: number }): Promise<StreamTestResult> {
    return request<StreamTestResult>("/api/cameras/test-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
  },

  /**
   * Validate stream reachability, protocol, and candidate endpoints
   */
  async validateStream(streamUrl: string): Promise<StreamValidationResult> {
    const params = new URLSearchParams({ stream_url: streamUrl });
    return request<StreamValidationResult>(`/api/stream/validate?${params.toString()}`);
  },

  /**
   * ANPR APIs
   */
  async scanAnprImage(file: File | Blob, cameraId: string = "CAM-UPLOAD"): Promise<AnprScanResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("camera_id", cameraId);
    return request<AnprScanResponse>("/api/anpr/image", {
      method: "POST",
      body: formData,
    });
  },

  async processAnprVideo(file: File | Blob, cameraId: string = "VIDEO-ANPR", stride: number = 15): Promise<AnprVideoResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("camera_id", cameraId);
    formData.append("stride", stride.toString());
    return request<AnprVideoResponse>("/api/anpr/video", {
      method: "POST",
      body: formData,
    });
  },

  async getAnprRecords(params?: { query?: string; cameraId?: string; limit?: number }): Promise<AnprRecord[]> {
    const qs = new URLSearchParams();
    if (params?.query) qs.append("query", params.query);
    if (params?.cameraId) qs.append("camera_id", params.cameraId);
    if (params?.limit) qs.append("limit", params.limit.toString());
    const queryString = qs.toString() ? `?${qs.toString()}` : "";
    return request<AnprRecord[]>(`/api/anpr/records${queryString}`);
  },

  async getAnprWatchlist(): Promise<WatchlistEntry[]> {
    return request<WatchlistEntry[]>("/api/anpr/watchlist");
  },

  async addAnprWatchlist(entry: { plate_number: string; reason?: string; severity?: string; vehicle_type?: string }): Promise<{ status: string; message: string; entry: WatchlistEntry }> {
    const formData = new FormData();
    formData.append("plate_number", entry.plate_number);
    if (entry.reason) formData.append("reason", entry.reason);
    if (entry.severity) formData.append("severity", entry.severity);
    if (entry.vehicle_type) formData.append("vehicle_type", entry.vehicle_type);
    return request("/api/anpr/watchlist", {
      method: "POST",
      body: formData,
    });
  },

  async deleteAnprWatchlist(plateNumber: string): Promise<{ status: string; message: string }> {
    return request(`/api/anpr/watchlist/${encodeURIComponent(plateNumber)}`, {
      method: "DELETE",
    });
  },

  /**
   * Facial Recognition APIs
   */
  async getEnrolledFaces(): Promise<{ success: boolean; count: number; persons: EnrolledPerson[] }> {
    return request<{ success: boolean; count: number; persons: EnrolledPerson[] }>("/api/faces");
  },

  async getFaceEngineStatus(): Promise<FaceEngineStatus> {
    return request<FaceEngineStatus>("/api/faces/status");
  },

  async getRecentFaceEvents(limit: number = 30): Promise<{ success: boolean; count: number; events: FaceEvent[] }> {
    return request<{ success: boolean; count: number; events: FaceEvent[] }>(`/api/faces/events?limit=${limit}`);
  },

  async registerFace(
    name: string,
    fileOrBase64: File | Blob | string,
    metadata?: {
      is_suspect?: boolean;
      threat_level?: string;
      category?: string;
      notes?: string;
    }
  ): Promise<RegisterFaceResponse> {
    const formData = new FormData();
    formData.append("name", name);
    if (metadata?.is_suspect !== undefined) {
      formData.append("is_suspect", String(metadata.is_suspect));
    }
    if (metadata?.threat_level) {
      formData.append("threat_level", metadata.threat_level);
    }
    if (metadata?.category) {
      formData.append("category", metadata.category);
    }
    if (metadata?.notes) {
      formData.append("notes", metadata.notes);
    }
    if (typeof fileOrBase64 === "string") {
      formData.append("image_base64", fileOrBase64);
    } else {
      formData.append("file", fileOrBase64);
    }
    return request<RegisterFaceResponse>("/api/faces/register", {
      method: "POST",
      body: formData,
    });
  },

  async updateFaceMetadata(
    personId: string,
    updates: {
      is_suspect?: boolean;
      threat_level?: string;
      category?: string;
      notes?: string;
    }
  ): Promise<{ success: boolean; person: EnrolledPerson; message: string }> {
    return request<{ success: boolean; person: EnrolledPerson; message: string }>(
      `/api/faces/${encodeURIComponent(personId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      }
    );
  },

  async addFaceSample(personId: string, fileOrBase64: File | Blob | string): Promise<RegisterFaceResponse> {
    const formData = new FormData();
    formData.append("person_id", personId);
    if (typeof fileOrBase64 === "string") {
      formData.append("image_base64", fileOrBase64);
    } else {
      formData.append("file", fileOrBase64);
    }
    return request<RegisterFaceResponse>("/api/faces/add-sample", {
      method: "POST",
      body: formData,
    });
  },

  async deleteFace(personId: string): Promise<{ success: boolean; message: string }> {
    return request<{ success: boolean; message: string }>(`/api/faces/${encodeURIComponent(personId)}`, {
      method: "DELETE",
    });
  },

  async scanFaceImage(
    fileOrBase64: File | Blob | string,
    minMatchScore: number = 0.40,
    minFaceSize: number = 40
  ): Promise<FaceScanResponse> {
    const formData = new FormData();
    formData.append("min_match_score", minMatchScore.toString());
    formData.append("min_face_size", minFaceSize.toString());
    if (typeof fileOrBase64 === "string") {
      formData.append("image_base64", fileOrBase64);
    } else {
      formData.append("file", fileOrBase64);
    }
    return request<FaceScanResponse>("/api/faces/scan", {
      method: "POST",
      body: formData,
    });
  },

  async stopFaceCamera(): Promise<{ success: boolean; released_cameras: number }> {
    return request<{ success: boolean; released_cameras: number }>("/api/faces/camera/stop", {
      method: "POST",
    });
  },

  /**
   * Security Alerts & Continuous Scanner APIs
   */
  async getAlerts(
    severity?: string,
    acknowledged?: boolean,
    cameraId?: string,
    limit: number = 100
  ): Promise<AlertItem[]> {
    const params = new URLSearchParams();
    if (severity && severity !== "All") params.append("severity", severity);
    if (acknowledged !== undefined) params.append("acknowledged", String(acknowledged));
    if (cameraId) params.append("camera_id", cameraId);
    params.append("limit", limit.toString());
    const query = params.toString() ? `?${params.toString()}` : "";
    return request<AlertItem[]>(`/api/alerts${query}`);
  },

  async createAlert(payload: {
    title: string;
    location: string;
    severity?: string;
    cameraId?: string;
    cameraName?: string;
    className?: string;
    confidence?: number;
    box?: number[];
    suspectName?: string;
    threatLevel?: string;
    category?: string;
    notes?: string;
  }): Promise<AlertItem> {
    return request<AlertItem>("/api/alerts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  async acknowledgeAlert(alertId: string): Promise<{ success: boolean; alert_id: string; acknowledged: boolean }> {
    return request<{ success: boolean; alert_id: string; acknowledged: boolean }>(
      `/api/alerts/${encodeURIComponent(alertId)}/acknowledge`,
      { method: "PATCH" }
    );
  },

  async clearAlerts(): Promise<{ success: boolean; message: string }> {
    return request<{ success: boolean; message: string }>("/api/alerts/clear", {
      method: "POST",
    });
  },

  async getScannerStatus(): Promise<ScannerStatus> {
    return request<ScannerStatus>("/api/alerts/scanner/status");
  },

  async startScanner(): Promise<{ success: boolean; message: string; monitored_cameras?: string[] }> {
    return request<{ success: boolean; message: string; monitored_cameras?: string[] }>(
      "/api/alerts/scanner/start",
      { method: "POST" }
    );
  },

  async stopScanner(): Promise<{ success: boolean; message: string }> {
    return request<{ success: boolean; message: string }>(
      "/api/alerts/scanner/stop",
      { method: "POST" }
    );
  },

  async getSuspectTrajectory(suspectName: string): Promise<SuspectTrajectory> {
    return request<SuspectTrajectory>(`/api/alerts/trajectory/${encodeURIComponent(suspectName)}`);
  },

  async getGlobalTraces(): Promise<{ status: string; total_active: number; traces: GlobalTraceItem[] }> {
    return request<{ status: string; total_active: number; traces: GlobalTraceItem[] }>("/api/tracking/global/traces");
  },

  async getGlobalTraceTrajectory(traceId: string): Promise<SuspectTrajectory> {
    return request<SuspectTrajectory>(`/api/tracking/global/traces/${encodeURIComponent(traceId)}`);
  },

  async resetGlobalTraces(): Promise<{ status: string; message: string }> {
    return request<{ status: string; message: string }>("/api/tracking/global/traces", { method: "DELETE" });
  },

  async dispatchQrt(
    alertId: string,
    unitName: string,
    notes?: string
  ): Promise<{ success: boolean; alert_id: string; dispatch: QrtDispatchRecord }> {
    return request<{ success: boolean; alert_id: string; dispatch: QrtDispatchRecord }>(
      `/api/alerts/${encodeURIComponent(alertId)}/dispatch`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ unitName, notes }),
      }
    );
  },

  // ── Geofences & Directional Tripwires ────────────────────────────────────
  async getGeofences(cameraId?: string): Promise<GeofencesResponse> {
    const query = cameraId ? `?camera_id=${encodeURIComponent(cameraId)}` : "";
    return request<GeofencesResponse>(`/api/geofences${query}`);
  },

  async createGeofenceZone(payload: CreateZonePayload): Promise<{ status: string; zone: GeofenceZone }> {
    return request<{ status: string; zone: GeofenceZone }>("/api/geofences/zones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  async updateGeofenceZone(
    zoneId: string,
    payload: Partial<CreateZonePayload>
  ): Promise<{ status: string; zone: GeofenceZone }> {
    return request<{ status: string; zone: GeofenceZone }>(`/api/geofences/zones/${encodeURIComponent(zoneId)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  async deleteGeofenceZone(zoneId: string): Promise<{ status: string; deleted_id: string }> {
    return request<{ status: string; deleted_id: string }>(`/api/geofences/zones/${encodeURIComponent(zoneId)}`, {
      method: "DELETE",
    });
  },

  async createTripwire(payload: CreateTripwirePayload): Promise<{ status: string; tripwire: DirectionalTripwire }> {
    return request<{ status: string; tripwire: DirectionalTripwire }>("/api/geofences/tripwires", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  async updateTripwire(
    wireId: string,
    payload: Partial<CreateTripwirePayload>
  ): Promise<{ status: string; tripwire: DirectionalTripwire }> {
    return request<{ status: string; tripwire: DirectionalTripwire }>(
      `/api/geofences/tripwires/${encodeURIComponent(wireId)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );
  },

  async deleteTripwire(wireId: string): Promise<{ status: string; deleted_id: string }> {
    return request<{ status: string; deleted_id: string }>(
      `/api/geofences/tripwires/${encodeURIComponent(wireId)}`,
      {
        method: "DELETE",
      }
    );
  },

  async evaluateGeofences(payload: {
    cameraId: string;
    tracks: Array<{ trackId: number; box: number[]; className?: string; confidence?: number; suspectName?: string }>;
    dispatchAlerts?: boolean;
  }): Promise<GeofenceEvaluateResponse> {
    return request<GeofenceEvaluateResponse>("/api/geofences/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  async resetGeofenceHistory(cameraId?: string): Promise<{ status: string; message: string }> {
    const query = cameraId ? `?camera_id=${encodeURIComponent(cameraId)}` : "";
    return request<{ status: string; message: string }>(`/api/geofences/reset-history${query}`, {
      method: "POST",
    });
  },
};


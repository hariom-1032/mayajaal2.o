export type GeofenceSeverity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

export type TripwireDirection = "FORWARD" | "REVERSE" | "BIDIRECTIONAL";

export type GeofenceAccessMode = "AUTHORIZED_PASS" | "ALL_ALERTS" | "SUSPECT_ONLY";

export interface GeofenceZone {
  id: string;
  name: string;
  cameraId: string;
  color: string;
  severity: GeofenceSeverity;
  polygon: [number, number][]; // Normalized [x, y] coordinates in [0.0, 1.0]
  targetClasses: string[];
  accessMode?: GeofenceAccessMode;
  enabled: boolean;
  cooldownSeconds: number;
  description?: string;
  createdAt?: string;
}

export interface DirectionalTripwire {
  id: string;
  name: string;
  cameraId: string;
  color: string;
  severity: GeofenceSeverity;
  p1: [number, number]; // Normalized start [x, y]
  p2: [number, number]; // Normalized end [x, y]
  direction: TripwireDirection;
  targetClasses: string[];
  accessMode?: GeofenceAccessMode;
  enabled: boolean;
  cooldownSeconds: number;
  description?: string;
  createdAt?: string;
}

export interface GeofencesResponse {
  zones: GeofenceZone[];
  tripwires: DirectionalTripwire[];
}

export interface GeofenceBreachEvent {
  type: "zone_breach" | "tripwire_crossing";
  alertType?: "suspect_breach" | "unknown_intrusion";
  zoneId?: string;
  wireId?: string;
  name: string;
  cameraId: string;
  trackId: number;
  className: string;
  confidence: number;
  suspectName?: string;
  personName?: string;
  identityStatus?: "authorized" | "suspect" | "unknown";
  footpoint: [number, number];
  directionCrossed?: string;
  requiredDirection?: string;
  severity: GeofenceSeverity;
  timestamp: string;
}

export interface GeofenceEvaluateResponse {
  status: string;
  cameraId: string;
  evaluatedTracks: number;
  breachesCount: number;
  breaches: GeofenceBreachEvent[];
}

export interface CreateZonePayload {
  name: string;
  cameraId: string;
  color: string;
  severity: GeofenceSeverity;
  polygon: [number, number][];
  targetClasses: string[];
  accessMode?: GeofenceAccessMode;
  enabled?: boolean;
  cooldownSeconds?: number;
  description?: string;
}

export interface CreateTripwirePayload {
  name: string;
  cameraId: string;
  color: string;
  severity: GeofenceSeverity;
  p1: [number, number];
  p2: [number, number];
  direction: TripwireDirection;
  targetClasses: string[];
  accessMode?: GeofenceAccessMode;
  enabled?: boolean;
  cooldownSeconds?: number;
  description?: string;
}

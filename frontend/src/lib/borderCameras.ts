import { Camera as CameraEntity } from "@/types/camera";

export const RTSP_CAMERA_TEMPLATE =
  "rtsp://admin:Admin123@142.67.45.150:554/cam/realmonitor?channel={CHANNEL}&subtype=0";

/**
 * Builds or updates the canonical RTSP URL for a given Dahua channel number.
 */
export function buildChannelRtspUrl(channel: number | string): string {
  const cleanCh = String(channel).trim();
  return `rtsp://admin:Admin123@142.67.45.150:554/cam/realmonitor?channel=${cleanCh}&subtype=0`;
}

/**
 * Updates an existing RTSP URL by replacing its channel parameter, or generates a new one.
 */
export function updateRtspUrlChannel(currentUrl: string | undefined, newChannel: number | string): string {
  const cleanCh = String(newChannel).trim();
  if (!currentUrl || !currentUrl.includes("channel=")) {
    return buildChannelRtspUrl(cleanCh);
  }
  return currentUrl.replace(/channel=\d+/i, `channel=${cleanCh}`);
}

/**
 * Extracts channel number from a given RTSP stream URL.
 */
export function extractChannelFromUrl(url?: string): number | null {
  if (!url) return null;
  const match = url.match(/channel=(\d+)/i);
  return match ? parseInt(match[1], 10) : null;
}

/**
 * Real active cameras (Only 5 core info fields: name, channel, type, status, lat/lng).
 * All fake sectors, dummy locations, fake resolutions, and mock bitrates have been removed.
 */
export const ALL_BORDER_CAMERAS: CameraEntity[] = [
  {
    id: "cam-ch-1",
    name: "CSD",
    channel: 1,
    type: "Optical 4K",
    status: "online",
    latitude: 23.15421,
    longitude: 72.885257,
    streamUrl: buildChannelRtspUrl(1),
  },
  {
    id: "cam-ch-2",
    name: "MAIN OFFICE OUTDOOR",
    channel: 2,
    type: "Panoramic",
    status: "online",
    latitude: 23.153804,
    longitude: 72.885263,
    streamUrl: buildChannelRtspUrl(2),
  },
  {
    id: "cam-ch-3",
    name: "MI ROOM",
    channel: 3,
    type: "PTZ 360",
    status: "alert",
    latitude: 23.153398,
    longitude: 72.88526,
    streamUrl: buildChannelRtspUrl(3),
  },
  {
    id: "cam-ch-4",
    name: "MAIN OFFICE INDOOR",
    channel: 4,
    type: "Thermal FLIR",
    status: "alert",
    latitude: 23.153357,
    longitude: 72.886619,
    streamUrl: buildChannelRtspUrl(4),
  },
  {
    id: "cam-ch-5",
    name: "SP HQ and D KOTE",
    channel: 5,
    type: "Optical 4K",
    status: "alert",
    latitude: 23.153343,
    longitude: 72.885683,
    streamUrl: buildChannelRtspUrl(5),
  },
  {
    id: "cam-ch-6",
    name: "KOTE REAR ENTRY & ARMOUR SHOP A KOTE",
    channel: 6,
    type: "Thermal FLIR",
    status: "alert",
    latitude: 32.748,
    longitude: 74.678,
    streamUrl: buildChannelRtspUrl(6),
  },
  {
    id: "cam-ch-7",
    name: "RP GATE",
    channel: 7,
    type: "Night Vision / IR",
    status: "degraded",
    latitude: 32.705,
    longitude: 74.685,
    streamUrl: buildChannelRtspUrl(7),
  },
  {
    id: "cam-ch-8",
    name: "RATION STORE",
    channel: 8,
    type: "Optical 4K",
    status: "online",
    latitude: 32.756,
    longitude: 74.684,
    streamUrl: buildChannelRtspUrl(8),
  },
  {
    id: "cam-ch-9",
    name: "PTZ CAM A COY",
    channel: 9,
    type: "PTZ 360",
    status: "online",
    latitude: 32.673,
    longitude: 74.688,
    streamUrl: buildChannelRtspUrl(9),
  },
  {
    id: "cam-ch-10",
    name: "C KOTE B KOTE & KADDI ROOM",
    channel: 10,
    type: "ANPR Dedicated",
    status: "online",
    latitude: 32.695,
    longitude: 74.667,
    streamUrl: buildChannelRtspUrl(10),
  },
];

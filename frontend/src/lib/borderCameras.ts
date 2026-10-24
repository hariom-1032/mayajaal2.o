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
    latitude: 23.229944,
    longitude: 72.708799,
    streamUrl: buildChannelRtspUrl(1),
  },
  {
    id: "cam-ch-2",
    name: "MAIN OFFICE OUTDOOR",
    channel: 2,
    type: "Panoramic",
    status: "online",
    latitude: 23.230034,
    longitude: 72.708789,
    streamUrl: buildChannelRtspUrl(2),
  },
  {
    id: "cam-ch-3",
    name: "MI ROOM",
    channel: 3,
    type: "PTZ 360",
    status: "alert",
    latitude: 23.229890,
    longitude: 72.708813,
    streamUrl: buildChannelRtspUrl(3),
  },
  {
    id: "cam-ch-4",
    name: "MAIN OFFICE INDOOR",
    channel: 4,
    type: "Thermal FLIR",
    status: "alert",
    latitude: 23.230102,
    longitude: 72.708823,
    streamUrl: buildChannelRtspUrl(4),
  },
  {
    id: "cam-ch-5",
    name: "SP HQ AND D KOTE",
    channel: 5,
    type: "Optical 4K",
    status: "alert",
    latitude: 23.230106,
    longitude: 72.707813,
    streamUrl: buildChannelRtspUrl(5),
  },
  {
    id: "cam-ch-6",
    name: "KOTE REAR ENTRY ARMOUR SHOP AND A KOTE",
    channel: 6,
    type: "Thermal FLIR",
    status: "alert",
    latitude: 23.230387,
    longitude: 72.707506,
    streamUrl: buildChannelRtspUrl(6),
  },
  {
    id: "cam-ch-7",
    name: "RP GATE",
    channel: 7,
    type: "Night Vision / IR",
    status: "degraded",
    latitude: 23.231270,
    longitude: 72.709568,
    streamUrl: buildChannelRtspUrl(7),
  },
  {
    id: "cam-ch-8",
    name: "RATION STORE",
    channel: 8,
    type: "Optical 4K",
    status: "online",
    latitude: 23.229333,
    longitude: 72.708813,
    streamUrl: buildChannelRtspUrl(8),
  },
  {
    id: "cam-ch-9",
    name: "PTZ CAM A COY",
    channel: 9,
    type: "PTZ 360",
    status: "online",
    latitude: 23.229135,
    longitude: 72.707216,
    streamUrl: buildChannelRtspUrl(9),
  },
  {
    id: "cam-ch-10",
    name: "C KOTE B KOTE AND KADDI ROOM",
    channel: 10,
    type: "ANPR Dedicated",
    status: "online",
    latitude: 23.230067,
    longitude: 72.707915,
    streamUrl: buildChannelRtspUrl(10),
  },
];

"use client";

import { useState, useEffect, useCallback } from "react";

export interface OperatorUser {
  id: string;
  name: string;
  rank: string;
  badgeNumber: string;
  sector: string;
  role: "Shift Commander" | "Tactical Operator" | "Intelligence Analyst" | "Defense Commander";
  clearanceLevel: string;
  serviceBranch: string;
  avatarInitials: string;
  authenticatedAt: string;
  expiresAt: string;
  rememberMe?: boolean;
  token?: string;
}




export const DEMO_OPERATORS: OperatorUser[] = [
  {
    id: "op-rathore",
    name: "Admin",
    rank: "System Administrator / Shift Commander",
    badgeNumber: "BSF-9482-KR",
    sector: "Sector-04 (BOP Alpha)",
    role: "Shift Commander",
    clearanceLevel: "Level 4 (Top Secret / Restricted)",
    serviceBranch: "Border Security Force (BSF)",
    avatarInitials: "AD",
    authenticatedAt: "2026-09-09T08:00:00Z",
    expiresAt: "2026-09-09T16:00:00Z",
  },
  {
    id: "op-sharma",
    name: "Sub-Insp. A. Sharma",
    rank: "Sub-Inspector / Surveillance",
    badgeNumber: "ITBP-4108-AS",
    sector: "Sector-05 (Sikkim Northern LAC)",
    role: "Tactical Operator",
    clearanceLevel: "Level 3 (Tactical Secret)",
    serviceBranch: "Indo-Tibetan Border Police (ITBP)",
    avatarInitials: "AS",
    authenticatedAt: "2026-09-09T08:30:00Z",
    expiresAt: "2026-09-09T16:30:00Z",
  },
  {
    id: "op-verma",
    name: "Capt. R. Verma",
    rank: "Operations Officer",
    badgeNumber: "RAW-7721-RV",
    sector: "Sector-01 (Punjab Western IB)",
    role: "Intelligence Analyst",
    clearanceLevel: "Level 4 (Quantum Grid Crypt)",
    serviceBranch: "Integrated Border Command (IBC)",
    avatarInitials: "RV",
    authenticatedAt: "2026-09-09T09:00:00Z",
    expiresAt: "2026-09-09T17:00:00Z",
  },
];

export const AUTHORIZED_PERSONNEL = [
  {
    ...DEMO_OPERATORS[0],
    passwords: ["MAATRIX2026", "BSF@2026", "COMMANDER", "ADMIN"],
  },
  {
    ...DEMO_OPERATORS[1],
    passwords: ["MAATRIX2026", "ITBP@2026", "OPERATOR"],
  },
  {
    ...DEMO_OPERATORS[2],
    passwords: ["MAATRIX2026", "RAW@2026", "ANALYST"],
  },
  {
    id: "op-commander",
    name: "Brig. V. S. Chauhan",
    rank: "Sector Commander",
    badgeNumber: "COMMANDER",
    sector: "Northern Unified HQ",
    role: "Shift Commander" as const,
    clearanceLevel: "Level 5 (Cosmic / Defense Command)",
    serviceBranch: "Integrated Defense Command (IDC)",
    avatarInitials: "VC",
    authenticatedAt: "2026-09-09T08:00:00Z",
    expiresAt: "2026-09-09T16:00:00Z",
    passwords: ["MAATRIX2026", "COMMANDER@2026", "ADMIN"],
  },
];

export const AUTH_STORAGE_KEY = "maatrix_operator_session_v1";
export const AUTH_COOKIE_NAME = "maatrix_session_token";
export const AUTH_EVENT_NAME = "maatrix_auth_change";

// Standard duty shift is 8 hours; rememberMe extends to 24 hours
export const SHIFT_DURATION_HOURS = 8;
export const REMEMBER_DURATION_HOURS = 24;

/**
 * Encodes session data safely into an HTTP-accessible cookie.
 */
export function setSessionCookie(token: string, expiresAt: Date) {
  if (typeof document === "undefined") return;
  const cookieStr = `${AUTH_COOKIE_NAME}=${encodeURIComponent(token)}; expires=${expiresAt.toUTCString()}; path=/; SameSite=Lax`;
  document.cookie = cookieStr;
}

/**
 * Clears the session cookie.
 */
export function clearSessionCookie() {
  if (typeof document === "undefined") return;
  document.cookie = `${AUTH_COOKIE_NAME}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax`;
}

/**
 * Checks whether a session has passed its expiration time.
 */
export function isSessionExpired(session: OperatorUser | null): boolean {
  if (!session || !session.expiresAt) return true;
  const expiryTime = new Date(session.expiresAt).getTime();
  if (isNaN(expiryTime)) return true;
  return Date.now() >= expiryTime;
}

/**
 * Calculates remaining shift time.
 */
export function getRemainingShiftTime(expiresAt?: string): {
  hours: number;
  minutes: number;
  seconds: number;
  isExpired: boolean;
  formatted: string;
} {
  if (!expiresAt) {
    return { hours: 0, minutes: 0, seconds: 0, isExpired: true, formatted: "Expired" };
  }
  const diffMs = new Date(expiresAt).getTime() - Date.now();
  if (diffMs <= 0) {
    return { hours: 0, minutes: 0, seconds: 0, isExpired: true, formatted: "Shift Expired" };
  }
  const totalSec = Math.floor(diffMs / 1000);
  const hours = Math.floor(totalSec / 3600);
  const minutes = Math.floor((totalSec % 3600) / 60);
  const seconds = totalSec % 60;
  const formatted = `${String(hours).padStart(2, "0")}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`;
  return { hours, minutes, seconds, isExpired: false, formatted };
}

/**
 * Generates an internal pseudo-token containing officer identity and expiry for Next.js middleware.
 */
export function createSessionToken(user: OperatorUser): string {
  const payload = {
    id: user.id,
    badge: user.badgeNumber,
    role: user.role,
    name: user.name,
    exp: new Date(user.expiresAt).getTime(),
  };
  try {
    return btoa(JSON.stringify(payload));
  } catch {
    return `maatrix_${user.id}_${new Date(user.expiresAt).getTime()}`;
  }
}

/**
 * Retrieves the stored operator session and validates its duty shift lifecycle.
 */
export function getStoredOperator(): OperatorUser | null {
  if (typeof window === "undefined") return null;
  try {
    let raw = localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) {
      // Migrate from legacy key if found
      raw = localStorage.getItem("mayajaal_operator_session_v1");
      if (raw) {
        localStorage.setItem(AUTH_STORAGE_KEY, raw);
        localStorage.removeItem("mayajaal_operator_session_v1");
      }
    }
    if (!raw) return null;

    const parsed: OperatorUser = JSON.parse(raw);

    // If session has expired, cleanly invalidate it
    if (isSessionExpired(parsed)) {
      localStorage.removeItem(AUTH_STORAGE_KEY);
      clearSessionCookie();
      return null;
    }

    // Auto-update legacy officer name to Admin
    if (parsed.name === "Insp. K. Rathore" || parsed.id === "op-rathore") {
      parsed.name = "Admin";
      if (parsed.avatarInitials === "KR") {
        parsed.avatarInitials = "AD";
      }
      try {
        localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(parsed));
      } catch {}
    }

    // Ensure session cookie is synced with valid localStorage session
    if (parsed.token && parsed.expiresAt) {
      setSessionCookie(parsed.token, new Date(parsed.expiresAt));
    }

    return parsed;
  } catch {
    return null;
  }
}

/**
 * Validates credentials strictly against authorized tactical defense personnel.
 */
export function validateCredentials(
  badgeInput: string,
  passwordInput: string
): { success: boolean; user?: OperatorUser; error?: string } {
  const cleanBadge = badgeInput.trim().toUpperCase();
  const cleanPass = passwordInput.trim();

  if (!cleanBadge) {
    return { success: false, error: "Please enter your Security Badge ID or Officer Code." };
  }
  if (!cleanPass) {
    return { success: false, error: "Please enter your Security Clearance Password." };
  }

  // Look for match in authorized roster
  const found = AUTHORIZED_PERSONNEL.find(
    (p) =>
      p.badgeNumber.toUpperCase() === cleanBadge ||
      p.id.toUpperCase() === cleanBadge ||
      (cleanBadge === "ADMIN" && (p.id === "op-rathore" || p.name === "Admin" || p.badgeNumber === "COMMANDER"))
  );

  if (!found) {
    return {
      success: false,
      error: `Badge ID "${cleanBadge}" is not recognized in the Border Surveillance Security Registry.`,
    };
  }

  // Validate password
  const isMatch = found.passwords.some(
    (pass) => pass.toLowerCase() === cleanPass.toLowerCase()
  );

  if (!isMatch) {
    return {
      success: false,
      error: "Authentication failed: Invalid security clearance password for this badge.",
    };
  }

  // Construct valid OperatorUser
  const { passwords: _, ...userProfile } = found;
  return {
    success: true,
    user: userProfile as OperatorUser,
  };
}

export function useAuth() {
  const [operator, setOperator] = useState<OperatorUser | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const syncState = useCallback(() => {
    const stored = getStoredOperator();
    setOperator((prev) => {
      if (!prev && !stored) return null;
      if (!prev || !stored) return stored;
      if (
        prev.name === stored.name &&
        prev.avatarInitials === stored.avatarInitials &&
        prev.badgeNumber === stored.badgeNumber &&
        prev.authenticatedAt === stored.authenticatedAt &&
        prev.expiresAt === stored.expiresAt &&
        prev.token === stored.token
      ) {
        return prev;
      }
      return stored;
    });
    setIsLoading(false);
  }, []);

  // Sync state from storage & cookies on mount
  useEffect(() => {
    syncState();

    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === AUTH_STORAGE_KEY) {
        syncState();
      }
    };

    const handleCustomAuthChange = () => {
      syncState();
    };

    window.addEventListener("storage", handleStorageChange);
    window.addEventListener(AUTH_EVENT_NAME, handleCustomAuthChange);

    // Periodic heartbeat to auto-expire duty shift if time elapses
    const interval = setInterval(() => {
      const current = getStoredOperator();
      if (!current) {
        setOperator((prev) => {
          if (prev) {
            window.dispatchEvent(new Event(AUTH_EVENT_NAME));
            return null;
          }
          return null;
        });
      }
    }, 15000);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
      window.removeEventListener(AUTH_EVENT_NAME, handleCustomAuthChange);
      clearInterval(interval);
    };
  }, [syncState]);

  /**
   * Initializes a formal operator duty shift session.
   */
  const login = useCallback(
    (user: OperatorUser, rememberMe: boolean = true) => {
      const now = new Date();
      const shiftHours = rememberMe ? REMEMBER_DURATION_HOURS : SHIFT_DURATION_HOURS;
      const expiryDate = new Date(now.getTime() + shiftHours * 60 * 60 * 1000);

      const sessionUser: OperatorUser = {
        ...user,
        authenticatedAt: now.toISOString(),
        expiresAt: expiryDate.toISOString(),
        rememberMe,
      };

      const token = createSessionToken(sessionUser);
      sessionUser.token = token;

      try {
        localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(sessionUser));
        setSessionCookie(token, expiryDate);
        setOperator(sessionUser);
        window.dispatchEvent(new Event(AUTH_EVENT_NAME));
      } catch (e) {
        console.error("Failed to store auth session:", e);
      }
    },
    []
  );

  /**
   * Quick-login helper for authorized demo officers.
   */
  const quickLogin = useCallback(
    (demoId: string) => {
      const match =
        DEMO_OPERATORS.find((o) => o.id === demoId || o.badgeNumber === demoId) ||
        DEMO_OPERATORS[0];
      login(match, true);
      return match;
    },
    [login]
  );

  /**
   * Immediately terminates the active shift session and clears all security tokens.
   */
  const logout = useCallback(() => {
    try {
      localStorage.removeItem(AUTH_STORAGE_KEY);
      clearSessionCookie();
      setOperator(null);
      window.dispatchEvent(new Event(AUTH_EVENT_NAME));
    } catch (e) {
      console.error("Failed to clear auth session:", e);
    }
  }, []);

  return {
    operator,
    isAuthenticated: Boolean(operator && !isSessionExpired(operator)),
    isLoading,
    login,
    quickLogin,
    logout,
  };
}

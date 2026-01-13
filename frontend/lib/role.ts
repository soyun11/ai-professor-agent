// frontend/lib/role.ts
export type UserRole = "professor" | "student";

const KEY = "aiagent:role";

export function getStoredRole(): UserRole {
  if (typeof window === "undefined") return "professor";
  const v = window.localStorage.getItem(KEY);
  return v === "student" ? "student" : "professor";
}

export function setStoredRole(role: UserRole) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(KEY, role);
}

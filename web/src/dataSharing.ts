interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

const STORAGE_KEY = "chess-scan:data-sharing-consent:v1";

export function loadDataSharingConsent(storage: StorageLike | null): boolean {
  if (!storage) return false;
  try {
    return storage.getItem(STORAGE_KEY) === "yes";
  } catch {
    return false;
  }
}

export function saveDataSharingConsent(
  storage: StorageLike | null,
  enabled: boolean,
): void {
  try {
    storage?.setItem(STORAGE_KEY, enabled ? "yes" : "no");
  } catch {
    // Scanning still works when storage is blocked in private or embedded contexts.
  }
}

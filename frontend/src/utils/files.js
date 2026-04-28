export function slugifyEventId(value) {
  const cleaned = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return cleaned || `event-${new Date().toISOString().slice(0, 10)}`;
}

export function formatBytes(bytes) {
  if (!bytes) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

export function buildFileId(file) {
  return [file.name, file.size, file.lastModified].join("-");
}

export function inferFileNameFromUrl(url, prefix = "match") {
  try {
    const parsed = new URL(url);
    const lastPart = parsed.pathname.split("/").pop();
    return lastPart || `${prefix}.jpg`;
  } catch {
    return `${prefix}.jpg`;
  }
}

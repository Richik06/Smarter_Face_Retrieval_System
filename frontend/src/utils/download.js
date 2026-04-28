export function saveBlob(blob, filename) {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 500);
}

export async function downloadUrlAsFile(url, filename) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Could not download image.");
  }
  const blob = await response.blob();
  saveBlob(blob, filename);
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function readError(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const payload = await response.json();
    return payload.detail || payload.error || "Request failed.";
  }
  return (await response.text()) || "Request failed.";
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

export function getCloudinarySignature(eventId, assetType = "events") {
  return request("/app/cloudinary/signature", {
    method: "POST",
    body: JSON.stringify({
      event_id: eventId,
      asset_type: assetType
    })
  });
}

export function processEventUrls({ eventId, imageUrls }) {
  return request("/app/process-event-urls", {
    method: "POST",
    body: JSON.stringify({
      event_id: eventId,
      image_urls: imageUrls
    })
  });
}

export function searchFaceUrl({ eventId, imageUrl }) {
  return request("/app/search-face-url", {
    method: "POST",
    body: JSON.stringify({
      event_id: eventId,
      image_url: imageUrl
    })
  });
}

export async function downloadAllMatches({ eventId, imageUrls }) {
  const response = await fetch(`${API_BASE}/app/download-all`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      event_id: eventId,
      image_urls: imageUrls
    })
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  return response.blob();
}

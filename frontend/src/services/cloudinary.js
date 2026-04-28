import { compressImage } from "../utils/compressImage";

export async function uploadImageToCloudinary({ file, signature, onProgress }) {
  const preparedFile = await compressImage(file, {
    maxBytes: 1_800_000,
    maxDimension: 2000,
    quality: 0.84
  });

  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("file", preparedFile);
    formData.append("api_key", signature.api_key);
    formData.append("timestamp", signature.timestamp);
    formData.append("signature", signature.signature);
    formData.append("folder", signature.folder);
    formData.append("use_filename", signature.use_filename);
    formData.append("unique_filename", signature.unique_filename);

    const request = new XMLHttpRequest();
    request.open("POST", signature.upload_url);

    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) {
        return;
      }
      const progress = Math.round((event.loaded / event.total) * 100);
      onProgress?.(progress);
    });

    request.addEventListener("load", () => {
      try {
        const payload = JSON.parse(request.responseText);
        if (request.status >= 400) {
          reject(new Error(payload?.error?.message || "Cloudinary upload failed."));
          return;
        }
        onProgress?.(100);
        resolve(payload);
      } catch {
        reject(new Error("Cloudinary returned an invalid response."));
      }
    });

    request.addEventListener("error", () => {
      reject(new Error("Network error while uploading to Cloudinary."));
    });

    request.send(formData);
  });
}

function injectTransformation(url, transformation) {
  if (!url || !url.includes("/upload/")) {
    return url;
  }
  return url.replace("/upload/", `/upload/${transformation}/`);
}

export function buildGalleryImageUrl(url) {
  return injectTransformation(url, "c_fill,g_auto,h_420,w_420/f_auto/q_auto");
}

export function buildFullImageUrl(url) {
  return injectTransformation(url, "c_limit,w_1600/f_auto/q_auto");
}

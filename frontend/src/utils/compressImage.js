function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Could not read the selected image."));
    };
    img.src = objectUrl;
  });
}

function renameWithExtension(name, type) {
  const extension = type === "image/png" ? ".png" : ".jpg";
  const base = name.replace(/\.[^.]+$/, "");
  return `${base}${extension}`;
}

export async function compressImage(file, options = {}) {
  if (!file.type.startsWith("image/")) {
    return file;
  }

  const settings = {
    maxBytes: options.maxBytes ?? 1_800_000,
    maxDimension: options.maxDimension ?? 2000,
    quality: options.quality ?? 0.84
  };

  const image = await loadImage(file);
  const largestDimension = Math.max(image.width, image.height);
  const scale = Math.min(1, settings.maxDimension / largestDimension);
  const shouldResize = scale < 1;
  const shouldCompress = file.size > settings.maxBytes;

  if (!shouldResize && !shouldCompress) {
    return file;
  }

  const canvas = document.createElement("canvas");
  canvas.width = Math.round(image.width * scale);
  canvas.height = Math.round(image.height * scale);

  const context = canvas.getContext("2d");
  if (!context) {
    return file;
  }
  context.drawImage(image, 0, 0, canvas.width, canvas.height);

  const outputType = file.type === "image/png" ? "image/png" : "image/jpeg";
  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, outputType, settings.quality)
  );

  if (!blob || blob.size >= file.size) {
    return file;
  }

  return new File([blob], renameWithExtension(file.name, outputType), {
    type: outputType,
    lastModified: file.lastModified
  });
}

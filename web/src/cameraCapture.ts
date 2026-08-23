const PREVIEW_MAX_DIMENSION = 720;

export const SOURCE_MAX_DIMENSION = 1600;

export interface CameraFrame {
  source: HTMLCanvasElement;
  preview: Blob;
}

export async function captureCameraFrame(video: HTMLVideoElement): Promise<CameraFrame> {
  // Detection and retained images share one draw, so accepted corners belong to the exact frame.
  const source = videoFrameCanvas(video, SOURCE_MAX_DIMENSION);
  const preview = resizedCanvas(source, PREVIEW_MAX_DIMENSION);
  return { source, preview: await canvasBlob(preview, 0.68) };
}

export async function captureSupplementalFrames(
  video: HTMLVideoElement,
  count: number,
): Promise<File[]> {
  const canvases: HTMLCanvasElement[] = [];
  for (let index = 0; index < count; index += 1) {
    await nextVideoFrame(video);
    canvases.push(videoFrameCanvas(video, SOURCE_MAX_DIMENSION));
  }
  const blobs = await Promise.all(canvases.map((canvas) => canvasBlob(canvas, 0.76)));
  return blobs.map((blob, index) => new File(
    [blob],
    `chess-board-frame-${index + 2}.jpg`,
    { type: "image/jpeg" },
  ));
}

export function videoFrameCanvas(
  video: HTMLVideoElement,
  maxDimension: number,
): HTMLCanvasElement {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  if (!sourceWidth || !sourceHeight) throw new Error("Camera has no frame yet");
  const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(sourceWidth * scale));
  canvas.height = Math.max(1, Math.round(sourceHeight * scale));
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas is unavailable");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

export function canvasBlob(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not capture camera frame"))),
      "image/jpeg",
      quality,
    );
  });
}

function resizedCanvas(source: HTMLCanvasElement, maxDimension: number): HTMLCanvasElement {
  const scale = Math.min(1, maxDimension / Math.max(source.width, source.height));
  if (scale === 1) return source;
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(source.width * scale));
  canvas.height = Math.max(1, Math.round(source.height * scale));
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas is unavailable");
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function nextVideoFrame(video: HTMLVideoElement): Promise<void> {
  return new Promise((resolve) => {
    let settled = false;
    let fallback = 0;
    const complete = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(fallback);
      resolve();
    };
    fallback = window.setTimeout(complete, 60);
    if (typeof video.requestVideoFrameCallback === "function") {
      video.requestVideoFrameCallback(() => complete());
    }
  });
}

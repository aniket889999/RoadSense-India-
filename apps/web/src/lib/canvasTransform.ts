/**
 * Pure coordinate transform helpers for RoadSense SiteOps ROI Canvas.
 * Implements deterministic contain / letterbox aspect-ratio preservation,
 * normalized image coordinate mapping, margin rejection, and devicePixelRatio scaling.
 */

export interface ViewportBox {
  x0: number;
  y0: number;
  width: number;
  height: number;
}

export interface Point2D {
  x: number;
  y: number;
}

/**
 * Compute the letterbox viewport box that fits an image inside the canvas
 * while preserving its native aspect ratio.
 */
export function computeContainViewport(
  canvasWidth: number,
  canvasHeight: number,
  imageWidth: number,
  imageHeight: number
): ViewportBox {
  if (canvasWidth <= 0 || canvasHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    return { x0: 0, y0: 0, width: Math.max(0, canvasWidth), height: Math.max(0, canvasHeight) };
  }

  const scale = Math.min(canvasWidth / imageWidth, canvasHeight / imageHeight);
  const w = imageWidth * scale;
  const h = imageHeight * scale;
  const x0 = (canvasWidth - w) / 2;
  const y0 = (canvasHeight - h) / 2;

  return { x0, y0, width: w, height: h };
}

/**
 * Transform a normalized coordinate [0.0, 1.0] relative to the reference image
 * into canvas rendering coordinates.
 */
export function imageNormalizedToCanvas(
  pt: Point2D,
  viewport: ViewportBox
): Point2D {
  return {
    x: viewport.x0 + pt.x * viewport.width,
    y: viewport.y0 + pt.y * viewport.height,
  };
}

/**
 * Transform canvas pointer coordinates into normalized image coordinates [0.0, 1.0].
 * Returns `null` if the pointer event is within the letterbox margin.
 */
export function canvasToImageNormalized(
  canvasPt: Point2D,
  viewport: ViewportBox,
  tolerancePixels: number = 0.5
): Point2D | null {
  const minX = viewport.x0 - tolerancePixels;
  const maxX = viewport.x0 + viewport.width + tolerancePixels;
  const minY = viewport.y0 - tolerancePixels;
  const maxY = viewport.y0 + viewport.height + tolerancePixels;

  if (canvasPt.x < minX || canvasPt.x > maxX || canvasPt.y < minY || canvasPt.y > maxY) {
    return null;
  }

  if (viewport.width <= 0 || viewport.height <= 0) {
    return null;
  }

  const normX = (canvasPt.x - viewport.x0) / viewport.width;
  const normY = (canvasPt.y - viewport.y0) / viewport.height;

  return {
    x: Math.max(0.0, Math.min(1.0, normX)),
    y: Math.max(0.0, Math.min(1.0, normY)),
  };
}

/**
 * Configure an HTML5 Canvas for sharp devicePixelRatio display
 * while keeping logical CSS coordinates aligned with pointer events.
 */
export function setupHiDPICanvas(
  canvas: HTMLCanvasElement,
  cssWidth: number,
  cssHeight: number,
  dpr: number = 1
): void {
  const effectiveDpr = Math.max(1, dpr);
  canvas.width = Math.round(cssWidth * effectiveDpr);
  canvas.height = Math.round(cssHeight * effectiveDpr);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;

  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.setTransform(effectiveDpr, 0, 0, effectiveDpr, 0, 0);
  }
}

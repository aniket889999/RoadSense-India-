import { AnprImageResult } from './anprTypes';

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function createReviewReport(result: AnprImageResult): Blob {
  return new Blob([JSON.stringify({
    report_schema: 'roadsense_anpr_review_v1',
    exported_at: new Date().toISOString(),
    human_verified: false,
    notice: 'Model candidates only. Check the original image. Format matches do not verify registration.',
    quality_notice: 'OpenCV quality measurements are advisory heuristics, not proof of readability or camera pose.',
    result,
  }, null, 2)], { type: 'application/json' });
}

export async function createAnnotatedImage(previewUrl: string, result: AnprImageResult): Promise<Blob> {
  const image = new Image();
  image.src = previewUrl;
  await image.decode();
  if (image.naturalWidth !== result.image_width || image.naturalHeight !== result.image_height) {
    throw new Error('Image orientation does not match the recognition result. Export the JSON report or try a normalized PNG.');
  }
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(360, result.image_width);
  const font = Math.max(12, Math.min(28, Math.round(canvas.width / 70)));
  const header = font * 3;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Annotated export is unavailable in this browser.');
  ctx.font = `${font}px sans-serif`;
  const lines: string[] = [];
  const entries = result.plates.length ? result.plates.map((plate, index) =>
    `#${index + 1} Candidate: ${plate.candidate_text || plate.normalized_text || '(unreadable)'}${plate.candidate_text ? ' (cleaned)' : ''} | ${plate.recognition_status} | NOT HUMAN VERIFIED`
  ) : [`${result.status}: no plate reading is available.`];
  for (const entry of entries) {
    let line = '';
    for (const character of entry) {
      if (line && ctx.measureText(line + character).width > canvas.width - 16) {
        lines.push(line);
        line = '';
      }
      line += character;
    }
    if (line) lines.push(line);
  }
  const lineHeight = Math.ceil(font * 1.4);
  const footer = 16 + lineHeight * lines.length;
  canvas.height = result.image_height + header + footer;
  if (canvas.width * canvas.height > 24_000_000) {
    throw new Error('The annotated image would be too large. Download the JSON review report instead.');
  }
  const sourceX = (canvas.width - result.image_width) / 2;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, sourceX, header, result.image_width, result.image_height);
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, canvas.width, header);
  ctx.fillStyle = '#fbbf24';
  ctx.font = `bold ${font}px sans-serif`;
  ctx.fillText('RoadSense ANPR — UNVERIFIED MODEL CANDIDATES', 8, font + 4);
  ctx.font = `${Math.max(10, font - 3)}px sans-serif`;
  ctx.fillText('Human review required • Complete readings and review cues are in the JSON report.', 8, font * 2 + 6);
  result.plates.forEach((plate, index) => {
    const box = plate.bbox;
    const color = plate.recognition_status === 'UNREADABLE' ? '#ef4444' : '#f59e0b';
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, canvas.width / 500);
    ctx.strokeRect(box.x1 + sourceX, box.y1 + header, box.x2 - box.x1, box.y2 - box.y1);
    ctx.font = `bold ${font}px sans-serif`;
    const label = `#${index + 1} ${plate.recognition_status}`;
    const labelWidth = Math.min(canvas.width, ctx.measureText(label).width + 12);
    const x = Math.max(0, Math.min(box.x1 + sourceX, canvas.width - labelWidth));
    const y = Math.max(header, box.y1 + header - font - 8);
    ctx.fillStyle = '#111827';
    ctx.fillRect(x, y, labelWidth, font + 8);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 4, y + font + 1);
  });
  ctx.fillStyle = '#f1f5f9';
  ctx.fillRect(0, header + result.image_height, canvas.width, footer);
  ctx.fillStyle = '#111827';
  ctx.font = `${font}px sans-serif`;
  lines.forEach((line, index) => ctx.fillText(line, 8, header + result.image_height + font + 8 + index * lineHeight));
  return new Promise((resolve, reject) => canvas.toBlob(
    blob => blob ? resolve(blob) : reject(new Error('The annotated image could not be encoded.')),
    'image/png',
  ));
}

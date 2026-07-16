import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const isPixelRect = (rect: Rect): boolean =>
  Number.isFinite(rect.x) &&
  Number.isFinite(rect.y) &&
  Number.isInteger(rect.width) &&
  Number.isInteger(rect.height) &&
  rect.width > 0 &&
  rect.height > 0;

/** Converts a document point to the export's canonical half-open integer pixel domain. */
export const documentToExportLocalSamPoint = (point: Vec2, exportRect: Rect, clampOutside = false): Vec2 | null => {
  if (!isPixelRect(exportRect) || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
    return null;
  }
  if (
    !clampOutside &&
    (point.x < exportRect.x ||
      point.x >= exportRect.x + exportRect.width ||
      point.y < exportRect.y ||
      point.y >= exportRect.y + exportRect.height)
  ) {
    return null;
  }
  return {
    x: clamp(Math.round(point.x - exportRect.x), 0, exportRect.width - 1),
    y: clamp(Math.round(point.y - exportRect.y), 0, exportRect.height - 1),
  };
};

/** Canonicalizes a document point while retaining document-space coordinates. */
export const canonicalizeDocumentSamPoint = (point: Vec2, exportRect: Rect, clampOutside = false): Vec2 | null => {
  const local = documentToExportLocalSamPoint(point, exportRect, clampOutside);
  return local ? { x: exportRect.x + local.x, y: exportRect.y + local.y } : null;
};

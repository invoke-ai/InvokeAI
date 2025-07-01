import type {
  CanvasBrushLineState,
  CanvasBrushLineWithPressureState,
  CanvasEraserLineState,
  CanvasEraserLineWithPressureState,
  CanvasRectState,
  CanvasImageState,
  Coordinate,
  Rect,
} from 'features/controlLayers/store/types';
import {
  transformMaskObject,
  clipMaskObjectToContainer,
  calculateMaskObjectBounds,
  convertTransformedToOriginal,
  type TransformedMaskObject,
} from './coordinateTransform';
import { maskObjectsToBitmap } from './bitmapToMaskObjects';

/**
 * Transforms mask objects relative to a bounding box container.
 * This adjusts all object coordinates to be relative to the bbox origin.
 * @param objects Array of mask objects to transform
 * @param bboxRect The bounding box to use as the container reference
 * @returns Array of transformed mask objects
 */
export function transformMaskObjectsRelativeToBbox(
  objects: (
    | CanvasBrushLineState
    | CanvasBrushLineWithPressureState
    | CanvasEraserLineState
    | CanvasEraserLineWithPressureState
    | CanvasRectState
    | CanvasImageState
  )[],
  bboxRect: Rect
): TransformedMaskObject[] {
  const transformedObjects: TransformedMaskObject[] = [];

  for (const obj of objects) {
    // Calculate the offset to make coordinates relative to the bbox
    const offset: Coordinate = {
      x: -bboxRect.x,
      y: -bboxRect.y,
    };

    const transformed = transformMaskObject(obj, offset);
    transformedObjects.push(transformed);
  }

  return transformedObjects;
}

/**
 * Clips all mask objects to the boundaries of a container rectangle.
 * @param objects Array of mask objects to clip
 * @param container The container rectangle to clip to
 * @returns Array of clipped mask objects (null values are filtered out)
 */
export function clipMaskObjectsToContainer(
  objects: TransformedMaskObject[],
  container: Rect
): TransformedMaskObject[] {
  return objects
    .map((obj) => clipMaskObjectToContainer(obj, container))
    .filter((obj): obj is TransformedMaskObject => obj !== null);
}

/**
 * Calculates the effective bounds of all mask objects.
 * @param objects Array of mask objects to calculate bounds for
 * @returns The bounding rectangle containing all objects, or null if no objects
 */
export function calculateMaskObjectsBounds(objects: TransformedMaskObject[]): Rect | null {
  if (objects.length === 0) {
    return null;
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const obj of objects) {
    const bounds = calculateMaskObjectBounds(obj);
    if (bounds) {
      minX = Math.min(minX, bounds.x);
      minY = Math.min(minY, bounds.y);
      maxX = Math.max(maxX, bounds.x + bounds.width);
      maxY = Math.max(maxY, bounds.y + bounds.height);
    }
  }

  if (minX === Infinity || minY === Infinity || maxX === -Infinity || maxY === -Infinity) {
    return null;
  }

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

/**
 * Calculates the bounding box of a consolidated mask by rendering it to a bitmap.
 * This provides the most accurate bounds by considering the actual rendered mask pixels.
 * @param objects Array of mask objects to calculate bounds for
 * @param canvasWidth Width of the canvas to render to
 * @param canvasHeight Height of the canvas to render to
 * @returns The bounding rectangle of the rendered mask, or null if no mask pixels
 */
export function calculateMaskBoundsFromBitmap(
  objects: TransformedMaskObject[],
  canvasWidth: number,
  canvasHeight: number
): Rect | null {
  if (objects.length === 0) {
    return null;
  }

  // Convert transformed objects back to original types for compatibility
  const originalObjects = objects.map(convertTransformedToOriginal);
  
  // Render the consolidated mask to a bitmap
  const bitmap = maskObjectsToBitmap(originalObjects, canvasWidth, canvasHeight);
  const { width, height, data } = bitmap;

  // Find the actual bounds of the rendered mask
  let maskMinX = width;
  let maskMinY = height;
  let maskMaxX = 0;
  let maskMaxY = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIndex = (y * width + x) * 4;
      const alpha = data[pixelIndex + 3] ?? 0;

      // If this pixel has any opacity, it's part of the mask
      if (alpha > 0) {
        maskMinX = Math.min(maskMinX, x);
        maskMinY = Math.min(maskMinY, y);
        maskMaxX = Math.max(maskMaxX, x);
        maskMaxY = Math.max(maskMaxY, y);
      }
    }
  }

  // If no mask pixels found, return null
  if (maskMinX >= maskMaxX || maskMinY >= maskMaxY) {
    return null;
  }

  return {
    x: maskMinX,
    y: maskMinY,
    width: maskMaxX - maskMinX + 1,
    height: maskMaxY - maskMinY + 1,
  };
}

/**
 * Inverts a mask by creating a new mask that covers the entire container except for the original mask areas.
 * @param objects Array of mask objects representing the original mask
 * @param container The container rectangle to invert within
 * @returns Array of mask objects representing the inverted mask
 */
export function invertMask(
  objects: TransformedMaskObject[],
  container: Rect
): TransformedMaskObject[] {
  // Create a rectangle that covers the entire container
  const fullCoverageRect: TransformedMaskObject = {
    id: 'inverted_mask_rect',
    type: 'rect',
    rect: {
      x: container.x,
      y: container.y,
      width: container.width,
      height: container.height,
    },
    color: { r: 255, g: 255, b: 255, a: 1 },
  };

  // For each original mask object, create an eraser line that removes it
  const eraserObjects: TransformedMaskObject[] = [];
  
  for (const obj of objects) {
    if (obj.type === 'rect') {
      // For rectangles, create an eraser rectangle
      const eraserRect: TransformedMaskObject = {
        id: `eraser_${obj.id}`,
        type: 'eraser_line',
        points: [
          obj.rect.x, obj.rect.y,
          obj.rect.x + obj.rect.width, obj.rect.y,
          obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height,
          obj.rect.x, obj.rect.y + obj.rect.height,
          obj.rect.x, obj.rect.y, // Close the rectangle
        ],
        strokeWidth: 1,
        clip: container,
      };
      eraserObjects.push(eraserRect);
    } else if (
      obj.type === 'brush_line' ||
      obj.type === 'brush_line_with_pressure' ||
      obj.type === 'eraser_line' ||
      obj.type === 'eraser_line_with_pressure'
    ) {
      // For lines, create an eraser line with the same points
      const eraserLine: TransformedMaskObject = {
        id: `eraser_${obj.id}`,
        type: 'eraser_line',
        points: [...obj.points],
        strokeWidth: obj.strokeWidth,
        clip: container,
      };
      eraserObjects.push(eraserLine);
    }
    // Note: Image objects are not handled in inversion as they're not commonly used in masks
  }

  return [fullCoverageRect, ...eraserObjects];
}

/**
 * Ensures all mask objects are clipped to the current bounding box boundaries.
 * This prevents masks from extending outside the bounding box after multiple inversions.
 * @param objects Array of mask objects to clip
 * @param bboxRect The bounding box to clip to
 * @returns Array of clipped mask objects
 */
export function ensureMaskObjectsWithinBbox(
  objects: TransformedMaskObject[],
  bboxRect: Rect
): TransformedMaskObject[] {
  return clipMaskObjectsToContainer(objects, bboxRect);
} 
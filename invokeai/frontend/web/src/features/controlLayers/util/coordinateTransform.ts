import type {
  CanvasBrushLineState,
  CanvasBrushLineWithPressureState,
  CanvasEraserLineState,
  CanvasEraserLineWithPressureState,
  CanvasImageState,
  CanvasRectState,
  Coordinate,
  Rect,
} from 'features/controlLayers/store/types';

/**
 * Type for mask objects with transformed coordinates.
 * This preserves the discriminated union structure while allowing coordinate transformations.
 */
export type TransformedMaskObject =
  | TransformedBrushLine
  | TransformedBrushLineWithPressure
  | TransformedEraserLine
  | TransformedEraserLineWithPressure
  | TransformedRect
  | TransformedImage;

export interface TransformedBrushLine {
  id: string;
  type: 'brush_line';
  points: number[];
  strokeWidth: number;
  color: { r: number; g: number; b: number; a: number };
  clip?: Rect | null;
}

export interface TransformedBrushLineWithPressure {
  id: string;
  type: 'brush_line_with_pressure';
  points: number[];
  strokeWidth: number;
  color: { r: number; g: number; b: number; a: number };
  clip?: Rect | null;
}

export interface TransformedEraserLine {
  id: string;
  type: 'eraser_line';
  points: number[];
  strokeWidth: number;
  clip?: Rect | null;
}

export interface TransformedEraserLineWithPressure {
  id: string;
  type: 'eraser_line_with_pressure';
  points: number[];
  strokeWidth: number;
  clip?: Rect | null;
}

export interface TransformedRect {
  id: string;
  type: 'rect';
  rect: Rect;
  color: { r: number; g: number; b: number; a: number };
}

export interface TransformedImage {
  id: string;
  type: 'image';
  image: { width: number; height: number; dataURL: string } | { width: number; height: number; image_name: string };
}

/**
 * Transforms a mask object by applying a coordinate offset.
 * @param obj The mask object to transform
 * @param offset The offset to apply to coordinates
 * @returns A new mask object with transformed coordinates
 */
export function transformMaskObject(
  obj:
    | CanvasBrushLineState
    | CanvasBrushLineWithPressureState
    | CanvasEraserLineState
    | CanvasEraserLineWithPressureState
    | CanvasRectState
    | CanvasImageState,
  offset: Coordinate
): TransformedMaskObject {
  switch (obj.type) {
    case 'brush_line':
      return {
        ...obj,
        points: transformPoints(obj.points, offset),
        clip: obj.clip ? transformRect(obj.clip, offset) : null,
      };
    case 'brush_line_with_pressure':
      return {
        ...obj,
        points: transformPoints(obj.points, offset),
        clip: obj.clip ? transformRect(obj.clip, offset) : null,
      };
    case 'eraser_line':
      return {
        ...obj,
        points: transformPoints(obj.points, offset),
        clip: obj.clip ? transformRect(obj.clip, offset) : null,
      };
    case 'eraser_line_with_pressure':
      return {
        ...obj,
        points: transformPoints(obj.points, offset),
        clip: obj.clip ? transformRect(obj.clip, offset) : null,
      };
    case 'rect':
      return {
        ...obj,
        rect: transformRect(obj.rect, offset),
      };
    case 'image':
      return {
        ...obj,
      };
  }
}

/**
 * Transforms an array of points by applying a coordinate offset.
 * @param points Array of numbers representing [x1, y1, x2, y2, ...]
 * @param offset The offset to apply
 * @returns New array with transformed coordinates
 */
export function transformPoints(points: number[], offset: Coordinate): number[] {
  const transformed: number[] = [];
  for (let i = 0; i < points.length; i += 2) {
    transformed.push((points[i] ?? 0) + offset.x);
    transformed.push((points[i + 1] ?? 0) + offset.y);
  }
  return transformed;
}

/**
 * Transforms a rectangle by applying a coordinate offset.
 * @param rect The rectangle to transform
 * @param offset The offset to apply
 * @returns New rectangle with transformed coordinates
 */
export function transformRect(rect: Rect, offset: Coordinate): Rect {
  return {
    x: rect.x + offset.x,
    y: rect.y + offset.y,
    width: rect.width,
    height: rect.height,
  };
}

/**
 * Clips a mask object to the boundaries of a container rectangle.
 * @param obj The mask object to clip
 * @param container The container rectangle to clip to
 * @returns A new mask object clipped to the container boundaries, or null if completely outside
 */
export function clipMaskObjectToContainer(obj: TransformedMaskObject, container: Rect): TransformedMaskObject | null {
  switch (obj.type) {
    case 'brush_line':
    case 'brush_line_with_pressure':
    case 'eraser_line':
    case 'eraser_line_with_pressure':
      return clipLineToContainer(obj, container);
    case 'rect':
      return clipRectToContainer(obj, container);
    case 'image':
      return clipImageToContainer(obj, container);
  }
}

/**
 * Clips a line object to container boundaries.
 */
function clipLineToContainer(
  obj:
    | TransformedBrushLine
    | TransformedBrushLineWithPressure
    | TransformedEraserLine
    | TransformedEraserLineWithPressure,
  container: Rect
): typeof obj | null {
  // For lines, we clip the points to the container boundaries
  const clippedPoints: number[] = [];

  for (let i = 0; i < obj.points.length; i += 2) {
    const x = obj.points[i] ?? 0;
    const y = obj.points[i + 1] ?? 0;

    // Clip coordinates to container boundaries
    const clippedX = Math.max(container.x, Math.min(container.x + container.width, x));
    const clippedY = Math.max(container.y, Math.min(container.y + container.height, y));

    clippedPoints.push(clippedX, clippedY);
  }

  // If no points remain, return null
  if (clippedPoints.length === 0) {
    return null;
  }

  return {
    ...obj,
    points: clippedPoints,
    clip: container,
  };
}

/**
 * Clips a rectangle object to container boundaries.
 */
function clipRectToContainer(obj: TransformedRect, container: Rect): TransformedRect | null {
  const rect = obj.rect;

  // Calculate intersection
  const left = Math.max(rect.x, container.x);
  const top = Math.max(rect.y, container.y);
  const right = Math.min(rect.x + rect.width, container.x + container.width);
  const bottom = Math.min(rect.y + rect.height, container.y + container.height);

  // If no intersection, return null
  if (left >= right || top >= bottom) {
    return null;
  }

  return {
    ...obj,
    rect: {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
    },
  };
}

/**
 * Clips an image object to container boundaries.
 */
function clipImageToContainer(obj: TransformedImage, _container: Rect): TransformedImage | null {
  // For images, we don't clip them - they remain as-is
  return obj;
}

/**
 * Calculates the effective bounds of a mask object.
 * @param obj The mask object to calculate bounds for
 * @returns The bounding rectangle, or null if the object has no effective bounds
 */
export function calculateMaskObjectBounds(obj: TransformedMaskObject): Rect | null {
  switch (obj.type) {
    case 'brush_line':
    case 'brush_line_with_pressure':
    case 'eraser_line':
    case 'eraser_line_with_pressure':
      return calculateLineBounds(obj);
    case 'rect':
      return obj.rect;
    case 'image':
      return calculateImageBounds(obj);
  }
}

/**
 * Calculates bounds for a line object.
 */
function calculateLineBounds(
  obj:
    | TransformedBrushLine
    | TransformedBrushLineWithPressure
    | TransformedEraserLine
    | TransformedEraserLineWithPressure
): Rect | null {
  if (obj.points.length < 2) {
    return null;
  }

  let minX = obj.points[0] ?? 0;
  let minY = obj.points[1] ?? 0;
  let maxX = minX;
  let maxY = minY;

  for (let i = 2; i < obj.points.length; i += 2) {
    const x = obj.points[i] ?? 0;
    const y = obj.points[i + 1] ?? 0;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  // Add stroke width to bounds
  const strokeRadius = obj.strokeWidth / 2;
  return {
    x: minX - strokeRadius,
    y: minY - strokeRadius,
    width: maxX - minX + obj.strokeWidth,
    height: maxY - minY + obj.strokeWidth,
  };
}

/**
 * Calculates bounds for an image object.
 */
function calculateImageBounds(obj: TransformedImage): Rect | null {
  return {
    x: 0,
    y: 0,
    width: obj.image.width,
    height: obj.image.height,
  };
}

/**
 * Converts a TransformedMaskObject back to its original mask object type.
 * This is needed for compatibility with functions that expect the original types.
 * @param obj The transformed mask object to convert back
 * @returns The original mask object type
 */
export function convertTransformedToOriginal(
  obj: TransformedMaskObject
):
  | CanvasBrushLineState
  | CanvasBrushLineWithPressureState
  | CanvasEraserLineState
  | CanvasEraserLineWithPressureState
  | CanvasRectState
  | CanvasImageState {
  switch (obj.type) {
    case 'brush_line':
      return {
        ...obj,
        clip: obj.clip ?? null,
      };
    case 'brush_line_with_pressure':
      return {
        ...obj,
        clip: obj.clip ?? null,
      };
    case 'eraser_line':
      return {
        ...obj,
        clip: obj.clip ?? null,
      };
    case 'eraser_line_with_pressure':
      return {
        ...obj,
        clip: obj.clip ?? null,
      };
    case 'rect':
      return obj;
    case 'image':
      return obj;
  }
}

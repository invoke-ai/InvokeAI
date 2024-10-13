import type { Selector, Store } from '@reduxjs/toolkit';
import { $authToken } from 'app/store/nanostores/authToken';
import type { CanvasEntityIdentifier, CanvasObjectState, Coordinate, Rect } from 'features/controlLayers/store/types';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Vector2d } from 'konva/lib/types';
import { clamp } from 'lodash-es';
import { customAlphabet } from 'nanoid';
import type { StrokeOptions } from 'perfect-freehand';
import getStroke from 'perfect-freehand';
import type { RgbColor } from 'react-colorful';
import { assert } from 'tsafe';

/**
 * Gets the scaled and floored cursor position on the stage. If the cursor is not currently over the stage, returns null.
 * @param stage The konva stage
 */
export const getScaledFlooredCursorPosition = (stage: Konva.Stage): Vector2d | null => {
  const pointerPosition = stage.getPointerPosition();
  const stageTransform = stage.getAbsoluteTransform().copy();
  if (!pointerPosition) {
    return null;
  }
  const scaledCursorPosition = stageTransform.invert().point(pointerPosition);
  return {
    x: Math.floor(scaledCursorPosition.x),
    y: Math.floor(scaledCursorPosition.y),
  };
};

/**
 * Gets the scaled cursor position on the stage. If the cursor is not currently over the stage, returns null.
 * @param stage The konva stage
 */
export const getScaledCursorPosition = (stage: Konva.Stage): Vector2d | null => {
  const pointerPosition = stage.getPointerPosition();
  const stageTransform = stage.getAbsoluteTransform().copy();
  if (!pointerPosition) {
    return null;
  }
  return stageTransform.invert().point(pointerPosition);
};

/**
 * Aligns a coordinate to the nearest integer. When the tool width is odd, an offset is added to align the edges
 * of the tool to the grid. Without this alignment, the edges of the tool will be 0.5px off.
 * @param coord The coordinate to align
 * @param toolWidth The width of the tool
 * @returns The aligned coordinate
 */
export const alignCoordForTool = (coord: Coordinate, toolWidth: number): Coordinate => {
  const roundedX = Math.round(coord.x);
  const roundedY = Math.round(coord.y);
  const deltaX = coord.x - roundedX;
  const deltaY = coord.y - roundedY;
  const offset = (toolWidth / 2) % 1;
  const point = {
    x: roundedX + Math.sign(deltaX) * offset,
    y: roundedY + Math.sign(deltaY) * offset,
  };
  return point;
};

/**
 * Offsets a point by the given offset. The offset is subtracted from the point.
 * @param coord The coordinate to offset
 * @param offset The offset to apply
 * @returns
 */
export const offsetCoord = (coord: Coordinate, offset: Coordinate): Coordinate => {
  return {
    x: coord.x - offset.x,
    y: coord.y - offset.y,
  };
};

/**
 * Snaps a position to the edge of the stage if within a threshold of the edge
 * @param pos The position to snap
 * @param stage The konva stage
 * @param snapPx The snap threshold in pixels
 */
export const snapPosToStage = (pos: Vector2d, stage: Konva.Stage, snapPx = 10): Vector2d => {
  const snappedPos = { ...pos };
  // Get the normalized threshold for snapping to the edge of the stage
  const thresholdX = snapPx / stage.scaleX();
  const thresholdY = snapPx / stage.scaleY();
  const stageWidth = stage.width() / stage.scaleX();
  const stageHeight = stage.height() / stage.scaleY();
  // Snap to the edge of the stage if within threshold
  if (pos.x - thresholdX < 0) {
    snappedPos.x = 0;
  } else if (pos.x + thresholdX > stageWidth) {
    snappedPos.x = Math.floor(stageWidth);
  }
  if (pos.y - thresholdY < 0) {
    snappedPos.y = 0;
  } else if (pos.y + thresholdY > stageHeight) {
    snappedPos.y = Math.floor(stageHeight);
  }
  return snappedPos;
};

export const floorCoord = (coord: Coordinate): Coordinate => {
  return {
    x: Math.floor(coord.x),
    y: Math.floor(coord.y),
  };
};

/**
 * Snaps a position to the edge of the given rect if within a threshold of the edge
 * @param pos The position to snap
 * @param rect The rect to snap to
 * @param threshold The snap threshold in pixels
 */
export const snapToRect = (pos: Vector2d, rect: Rect, threshold = 10): Vector2d => {
  const snappedPos = { ...pos };
  // Snap to the edge of the rect if within threshold
  if (pos.x - threshold < rect.x) {
    snappedPos.x = rect.x;
  } else if (pos.x + threshold > rect.x + rect.width) {
    snappedPos.x = rect.x + rect.width;
  }
  if (pos.y - threshold < rect.y) {
    snappedPos.y = rect.y;
  } else if (pos.y + threshold > rect.y + rect.height) {
    snappedPos.y = rect.y + rect.height;
  }
  return snappedPos;
};

/**
 * Checks if the left mouse button is currently pressed
 * @param e The konva event
 */
export const getIsMouseDown = (e: KonvaEventObject<MouseEvent>): boolean => e.evt.buttons === 1;

/**
 * Checks if the stage is currently focused
 * @param stage The konva stage
 */
export const getIsFocused = (stage: Konva.Stage): boolean => stage.container().contains(document.activeElement);

/**
 * Gets the last point of a line as a coordinate.
 * @param points An array of numbers representing points as [x1, y1, x2, y2, ...]
 * @returns The last point of the line as a coordinate, or null if the line has less than 1 point
 */
export const getLastPointOfLine = (points: number[]): Coordinate | null => {
  if (points.length < 2) {
    return null;
  }
  const x = points.at(-2);
  const y = points.at(-1);
  if (x === undefined || y === undefined) {
    return null;
  }
  return { x, y };
};

/**
 * Gets the last point of a line as a coordinate.
 * @param points An array of numbers representing points as [x1, y1, x2, y2, ...]
 * @returns The last point of the line as a coordinate, or null if the line has less than 1 point
 */
export const getLastPointOfLineWithPressure = (points: number[]): CoordinateWithPressure | null => {
  if (points.length < 3) {
    return null;
  }
  const x = points.at(-3);
  const y = points.at(-2);
  const pressure = points.at(-1);
  if (x === undefined || y === undefined || pressure === undefined) {
    return null;
  }
  return { x, y, pressure };
};

export function getIsPrimaryMouseDown(e: KonvaEventObject<MouseEvent>) {
  return e.evt.buttons === 1;
}

/**
 * Calculates the new brush size based on the current brush size and the wheel delta from a mouse wheel event.
 * @param brushSize The current brush size
 * @param delta The wheel delta
 * @returns
 */
export const calculateNewBrushSizeFromWheelDelta = (brushSize: number, delta: number) => {
  // This equation was derived by fitting a curve to the desired brush sizes and deltas
  // see https://github.com/invoke-ai/InvokeAI/pull/5542#issuecomment-1915847565
  const targetDelta = Math.sign(delta) * 0.7363 * Math.pow(1.0394, brushSize);
  // This needs to be clamped to prevent the delta from getting too large
  const finalDelta = clamp(targetDelta, -20, 20);
  // The new brush size is also clamped to prevent it from getting too large or small
  const newBrushSize = clamp(brushSize + finalDelta, 1, 500);

  return newBrushSize;
};

/**
 * Checks if a candidate point is at least `minDistance` away from the last point. If there is no last point, returns true.
 * @param candidatePoint The candidate point
 * @param lastPoint The last point
 * @param minDistance The minimum distance between points
 * @returns Whether the candidate point is at least `minDistance` away from the last point
 */
export const isDistanceMoreThanMin = (
  candidatePoint: Coordinate,
  lastPoint: Coordinate | null,
  minDistance: number
): boolean => {
  if (!lastPoint) {
    return true;
  }

  return Math.hypot(lastPoint.x - candidatePoint.x, lastPoint.y - candidatePoint.y) >= minDistance;
};

/**
 * Simple util to map an object to its id property. Serves as a minor optimization to avoid recreating a map callback
 * every time we need to map an object to its id, which happens very often.
 * @param object The object with an `id` property
 * @returns The object's id property
 */
export const mapId = (object: { id: string }): string => object.id;

/**
 * Convert a Blob to a data URL.
 */
export const blobToDataURL = (blob: Blob): Promise<string> => {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (_e) => resolve(reader.result as string);
    reader.onerror = (_e) => reject(reader.error);
    reader.onabort = (_e) => reject(new Error('Read aborted'));
    reader.readAsDataURL(blob);
  });
};

/**
 * Convert an ImageData object to a data URL.
 */
export function imageDataToDataURL(imageData: ImageData): string {
  const { width, height } = imageData;

  // Create a canvas to transfer the ImageData to
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  // Draw the ImageData onto the canvas
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Unable to get canvas context');
  }
  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(imageData, 0, 0);

  // Convert the canvas to a data URL (base64)
  return canvas.toDataURL();
}

export function imageDataToBlob(imageData: ImageData): Promise<Blob | null> {
  const w = imageData.width;
  const h = imageData.height;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    return Promise.resolve(null);
  }

  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(imageData, 0, 0);

  return new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve);
  });
}

/**
 * Download a Blob as a file
 */
export const downloadBlob = (blob: Blob, fileName: string) => {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  a.remove();
};

/**
 * Gets an ImageData object from an image dataURL by drawing it to a canvas.
 */
export const dataURLToImageData = (dataURL: string, width: number, height: number): Promise<ImageData> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      canvas.remove();
      reject('Unable to get context');
      return;
    }

    ctx.imageSmoothingEnabled = false;

    const image = new Image();
    image.onload = function () {
      ctx.drawImage(image, 0, 0);
      canvas.remove();
      resolve(ctx.getImageData(0, 0, width, height));
    };

    image.onerror = function (e) {
      canvas.remove();
      reject(e);
    };

    image.crossOrigin = $authToken.get() ? 'use-credentials' : 'anonymous';
    image.src = dataURL;
  });
};

export const konvaNodeToCanvas = (arg: { node: Konva.Node; rect?: Rect; bg?: string }): HTMLCanvasElement => {
  const { node, rect, bg } = arg;
  const canvas = node.toCanvas({ ...(rect ?? {}), imageSmoothingEnabled: false, pixelRatio: 1 });

  if (!bg) {
    return canvas;
  }

  // We need to draw the canvas onto a new canvas with the specified background color
  const bgCanvas = document.createElement('canvas');
  bgCanvas.width = canvas.width;
  bgCanvas.height = canvas.height;
  const bgCtx = bgCanvas.getContext('2d');
  assert(bgCtx !== null, 'bgCtx is null');
  bgCtx.imageSmoothingEnabled = false;
  bgCtx.fillStyle = bg;
  bgCtx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);
  bgCtx.drawImage(canvas, 0, 0);
  return bgCanvas;
};

/**
 * Converts a HTMLCanvasElement to a Blob
 * @param canvas - The canvas to convert to a Blob
 * @returns A Promise that resolves to the Blob, or rejects if the conversion fails
 */
export const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject('Failed to convert canvas to blob');
      } else {
        resolve(blob);
      }
    });
  });
};

export const canvasToImageData = (canvas: HTMLCanvasElement): ImageData => {
  const ctx = canvas.getContext('2d');
  assert(ctx, 'ctx is null');
  ctx.imageSmoothingEnabled = false;
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
};

/**
 * Converts a Konva node to an ImageData object
 * @param node - The Konva node to convert to an ImageData object
 * @param rect - The bounding box to crop to
 * @returns A Promise that resolves with ImageData object of the node cropped to the bounding box
 */
export const konvaNodeToImageData = (arg: { node: Konva.Node; rect?: Rect; bg?: string }): ImageData => {
  const { node, rect, bg } = arg;
  const canvas = konvaNodeToCanvas({ node, rect, bg });
  return canvasToImageData(canvas);
};

/**
 * Converts a Konva node to a Blob
 * @param node - The Konva node to convert to a Blob
 * @param rect - The bounding box to crop to
 * @returns A Promise that resolves to the Blob or null,
 */
export const konvaNodeToBlob = (arg: { node: Konva.Node; rect?: Rect; bg?: string }): Promise<Blob> => {
  const { node, rect, bg } = arg;
  const canvas = konvaNodeToCanvas({ node, rect, bg });
  return canvasToBlob(canvas);
};

export const previewBlob = (blob: Blob, label?: string) => {
  const url = URL.createObjectURL(blob);
  const w = window.open('');
  if (!w) {
    return;
  }
  if (label) {
    w.document.write(label);
    w.document.write('</br>');
  }
  w.document.write(`<img src="${url}" style="border: 1px solid red;" />`);
};

export type Transparency = 'FULLY_TRANSPARENT' | 'PARTIALLY_TRANSPARENT' | 'OPAQUE';
export function getImageDataTransparency(imageData: ImageData): Transparency {
  let isFullyTransparent = true;
  let isPartiallyTransparent = false;
  const len = imageData.data.length;
  for (let i = 3; i < len; i += 4) {
    if (imageData.data[i] !== 0) {
      isFullyTransparent = false;
    } else {
      isPartiallyTransparent = true;
    }
    if (!isFullyTransparent && isPartiallyTransparent) {
      return 'PARTIALLY_TRANSPARENT';
    }
  }
  if (isFullyTransparent) {
    return 'FULLY_TRANSPARENT';
  }
  if (isPartiallyTransparent) {
    return 'PARTIALLY_TRANSPARENT';
  }
  return 'OPAQUE';
}

/**
 * Loads an image from a URL and returns a promise that resolves with the loaded image element.
 * @param src The image source URL
 * @returns A promise that resolves with the loaded image element
 */
export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const imageElement = new Image();
    imageElement.onload = () => resolve(imageElement);
    imageElement.onerror = (error) => reject(error);
    imageElement.crossOrigin = $authToken.get() ? 'use-credentials' : 'anonymous';
    imageElement.src = src;
  });
}

/**
 * Generates a random alphanumeric string of length 10. Probably not secure at all.
 */
export const nanoid = customAlphabet('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', 10);

export function getPrefixedId(
  prefix: CanvasEntityIdentifier['type'] | CanvasObjectState['type'] | (string & Record<never, never>)
): string {
  return `${prefix}:${nanoid()}`;
}

export const getEmptyRect = (): Rect => {
  return { x: 0, y: 0, width: 0, height: 0 };
};

export function snapToNearest(value: number, candidateValues: number[], threshold: number = Infinity): number {
  let closest = value;
  let minDiff = Number.MAX_VALUE;

  for (const candidate of candidateValues) {
    const diff = Math.abs(value - candidate);
    if (diff < minDiff && diff <= threshold) {
      minDiff = diff;
      closest = candidate;
    }
  }

  return closest;
}

/**
 * Gets the union of any number of rects.
 * @params rects The rects to union
 * @returns The union of the two rects
 */
export const getRectUnion = (...rects: Rect[]): Rect => {
  const firstRect = rects.shift();

  if (!firstRect) {
    return getEmptyRect();
  }

  const rect = rects.reduce<Rect>((acc, r) => {
    const x = Math.min(acc.x, r.x);
    const y = Math.min(acc.y, r.y);
    const width = Math.max(acc.x + acc.width, r.x + r.width) - x;
    const height = Math.max(acc.y + acc.height, r.y + r.height) - y;
    return { x, y, width, height };
  }, firstRect);

  return rect;
};

/**
 * Gets the intersection of any number of rects.
 * @params rects The rects to intersect
 * @returns The intersection of the rects, or an empty rect if no intersection exists
 */
export const getRectIntersection = (...rects: Rect[]): Rect => {
  const firstRect = rects.shift();

  if (!firstRect) {
    return getEmptyRect();
  }

  const rect = rects.reduce<Rect>((acc, r) => {
    const x = Math.max(acc.x, r.x);
    const y = Math.max(acc.y, r.y);
    const width = Math.min(acc.x + acc.width, r.x + r.width) - x;
    const height = Math.min(acc.y + acc.height, r.y + r.height) - y;

    // We continue even if width or height is negative, and check at the end
    return { x, y, width, height };
  }, firstRect);

  // Final check to ensure positive width and height, else return empty rect
  if (rect.width < 0 || rect.height < 0) {
    return getEmptyRect();
  }

  return rect || getEmptyRect();
};

/**
 * Asserts that the value is never reached. Used for exhaustive checks in switch statements or conditional logic to ensure
 * that all possible values are handled.
 * @param value The value that should never be reached
 * @throws An error with the value that was not handled
 */
export const exhaustiveCheck = (value: never): never => {
  assert(false, `Unhandled value: ${value}`);
};

type CoordinateWithPressure = {
  x: number;
  y: number;
  pressure: number;
};
export const getLastPointOfLastLineWithPressure = (
  objects: CanvasObjectState[],
  type: 'brush_line_with_pressure' | 'eraser_line_with_pressure'
): CoordinateWithPressure | null => {
  const lastObject = objects.at(-1);
  if (!lastObject) {
    return null;
  }

  if (lastObject.type === type) {
    return getLastPointOfLineWithPressure(lastObject.points);
  }

  return null;
};

export const getLastPointOfLastLine = (
  objects: CanvasObjectState[],
  type: 'brush_line' | 'eraser_line'
): Coordinate | null => {
  const lastObject = objects.at(-1);
  if (!lastObject) {
    return null;
  }

  if (lastObject.type === type) {
    return getLastPointOfLine(lastObject.points);
  }

  return null;
};

export type SubscriptionHandler<T> = (value: T, prevValue: T) => void;

export const createReduxSubscription = <T, U>(
  store: Store<T>,
  selector: Selector<T, U>,
  handler: SubscriptionHandler<U>
) => {
  let prevValue: U = selector(store.getState());
  const unsubscribe = store.subscribe(() => {
    const value = selector(store.getState());
    if (value !== prevValue) {
      handler(value, prevValue);
      prevValue = value;
    }
  });

  return unsubscribe;
};

export const getKonvaNodeDebugAttrs = (node: Konva.Node) => {
  return {
    x: node.x(),
    y: node.y(),
    width: node.width(),
    height: node.height(),
    scaleX: node.scaleX(),
    scaleY: node.scaleY(),
    offsetX: node.offsetX(),
    offsetY: node.offsetY(),
    rotation: node.rotation(),
    isCached: node.isCached(),
    visible: node.visible(),
    listening: node.listening(),
  };
};

const average = (a: number, b: number) => (a + b) / 2;

function getSvgPathFromStroke(points: number[][], closed = true) {
  const len = points.length;

  if (len < 4) {
    return '';
  }

  let a = points[0] as number[];
  let b = points[1] as number[];
  const c = points[2] as number[];

  let result = `M${a[0]!.toFixed(2)},${a[1]!.toFixed(2)} Q${b[0]!.toFixed(
    2
  )},${b[1]!.toFixed(2)} ${average(b[0]!, c[0]!).toFixed(2)},${average(b[1]!, c[1]!).toFixed(2)} T`;

  for (let i = 2, max = len - 1; i < max; i++) {
    a = points[i]!;
    b = points[i + 1]!;
    result += `${average(a[0]!, b[0]!).toFixed(2)},${average(a[1]!, b[1]!).toFixed(2)} `;
  }

  if (closed) {
    result += 'Z';
  }

  return result;
}

export const getSVGPathDataFromPoints = (points: number[], options?: StrokeOptions): string => {
  const chunked: [number, number, number][] = [];
  for (let i = 0; i < points.length; i += 3) {
    chunked.push([points[i]!, points[i + 1]!, points[i + 2]!]);
  }
  return getSvgPathFromStroke(getStroke(chunked, options));
};

export const getPointerType = (e: KonvaEventObject<PointerEvent>): 'mouse' | 'pen' | 'touch' => {
  if (e.evt.pointerType === 'mouse') {
    return 'mouse';
  }

  if (e.evt.pointerType === 'pen') {
    return 'pen';
  }

  return 'touch';
};

/**
 * Gets the color at the given coordinate on the stage.
 * @param stage The konva stage.
 * @param coord The coordinate to get the color at. This must be the _absolute_ coordinate on the stage.
 * @returns The color under the coordinate, or null if there was a problem getting the color.
 */
export const getColorAtCoordinate = (stage: Konva.Stage, coord: Coordinate): RgbColor | null => {
  const ctx = stage
    .toCanvas({ x: coord.x, y: coord.y, width: 1, height: 1, imageSmoothingEnabled: false })
    .getContext('2d');

  if (!ctx) {
    return null;
  }

  const [r, g, b, _a] = ctx.getImageData(0, 0, 1, 1).data;

  if (r === undefined || g === undefined || b === undefined) {
    return null;
  }

  return { r, g, b };
};

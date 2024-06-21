import {
  CA_LAYER_NAME,
  INPAINT_MASK_LAYER_NAME,
  RASTER_LAYER_BRUSH_LINE_NAME,
  RASTER_LAYER_ERASER_LINE_NAME,
  RASTER_LAYER_IMAGE_NAME,
  RASTER_LAYER_NAME,
  RASTER_LAYER_RECT_SHAPE_NAME,
  RG_LAYER_BRUSH_LINE_NAME,
  RG_LAYER_ERASER_LINE_NAME,
  RG_LAYER_NAME,
  RG_LAYER_RECT_SHAPE_NAME,
} from 'features/controlLayers/konva/naming';
import type { Rect, RgbaColor } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Vector2d } from 'konva/lib/types';
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
 * Simple util to map an object to its id property. Serves as a minor optimization to avoid recreating a map callback
 * every time we need to map an object to its id, which happens very often.
 * @param object The object with an `id` property
 * @returns The object's id property
 */
export const mapId = (object: { id: string }): string => object.id;

/**
 * Konva selection callback to select all renderable layers. This includes RG, CA II and Raster layers.
 * This can be provided to the `find` or `findOne` konva node methods.
 */
export const selectRenderableLayers = (node: Konva.Node): boolean =>
  node.name() === RG_LAYER_NAME ||
  node.name() === CA_LAYER_NAME ||
  node.name() === RASTER_LAYER_NAME ||
  node.name() === INPAINT_MASK_LAYER_NAME;

/**
 * Konva selection callback to select RG mask objects. This includes lines and rects.
 * This can be provided to the `find` or `findOne` konva node methods.
 */
export const selectVectorMaskObjects = (node: Konva.Node): boolean =>
  node.name() === RG_LAYER_BRUSH_LINE_NAME ||
  node.name() === RG_LAYER_ERASER_LINE_NAME ||
  node.name() === RG_LAYER_RECT_SHAPE_NAME;

/**
 * Konva selection callback to select raster layer objects. This includes lines and rects.
 * This can be provided to the `find` or `findOne` konva node methods.
 */
export const selectRasterObjects = (node: Konva.Node): boolean =>
  node.name() === RASTER_LAYER_BRUSH_LINE_NAME ||
  node.name() === RASTER_LAYER_ERASER_LINE_NAME ||
  node.name() === RASTER_LAYER_RECT_SHAPE_NAME ||
  node.name() === RASTER_LAYER_IMAGE_NAME;

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
  ctx.putImageData(imageData, 0, 0);

  // Convert the canvas to a data URL (base64)
  return canvas.toDataURL();
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
 * Gets a Blob from a HTMLCanvasElement.
 */
export const canvasToBlob = async (canvas: HTMLCanvasElement): Promise<Blob> => {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject('Unable to create Blob');
    });
  });
};

/**
 * Gets an ImageData object from an image dataURL by drawing it to a canvas.
 */
export const dataURLToImageData = async (dataURL: string, width: number, height: number): Promise<ImageData> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const image = new Image();

    if (!ctx) {
      canvas.remove();
      reject('Unable to get context');
      return;
    }

    image.onload = function () {
      ctx.drawImage(image, 0, 0);
      canvas.remove();
      resolve(ctx.getImageData(0, 0, width, height));
    };

    image.src = dataURL;
  });
};

/**
 * Converts a Konva node to a Blob
 * @param node - The Konva node to convert to a Blob
 * @param bbox - The bounding box to crop to
 * @returns A Promise that resolves with Blob of the node cropped to the bounding box
 */
export const konvaNodeToBlob = async (node: Konva.Node, bbox?: Rect): Promise<Blob> => {
  return await new Promise<Blob>((resolve) => {
    node.toBlob({
      callback: (blob) => {
        assert(blob, 'Blob is null');
        resolve(blob);
      },
      ...(bbox ?? {}),
    });
  });
};

/**
 * Converts a Konva node to an ImageData object
 * @param node - The Konva node to convert to an ImageData object
 * @param bbox - The bounding box to crop to
 * @returns A Promise that resolves with ImageData object of the node cropped to the bounding box
 */
export const konvaNodeToImageData = (node: Konva.Node, bbox?: Rect): ImageData => {
  // get a dataURL of the bbox'd region
  const canvas = node.toCanvas({ ...(bbox ?? {}) });
  const ctx = canvas.getContext('2d');
  assert(ctx, 'ctx is null');
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
};

/**
 * Gets the pixel under the cursor on the stage, or null if the cursor is not over the stage.
 * @param stage The konva stage
 */
export const getPixelUnderCursor = (stage: Konva.Stage): RgbaColor | null => {
  const cursorPos = stage.getPointerPosition();
  const pixelRatio = Konva.pixelRatio;
  if (!cursorPos) {
    return null;
  }
  const ctx = stage.toCanvas().getContext('2d');

  if (!ctx) {
    return null;
  }
  const [r, g, b, a] = ctx.getImageData(cursorPos.x * pixelRatio, cursorPos.y * pixelRatio, 1, 1).data;

  if (r === undefined || g === undefined || b === undefined || a === undefined) {
    return null;
  }

  return { r, g, b, a };
};

export const previewBlob = async (blob: Blob, label?: string) => {
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

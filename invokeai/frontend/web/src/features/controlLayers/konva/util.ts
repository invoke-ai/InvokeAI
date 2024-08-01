import { getImageDataTransparency } from 'common/util/arrayBuffer';
import { CanvasLayer } from 'features/controlLayers/konva/CanvasLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { GenerationMode, Rect, RenderableObject, RgbaColor } from 'features/controlLayers/store/types';
import { isValidLayer } from 'features/nodes/util/graph/generation/addLayers';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Vector2d } from 'konva/lib/types';
import { customAlphabet } from 'nanoid';
import type { WritableAtom } from 'nanostores';
import type { ImageDTO } from 'services/api/types';
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

export const konvaNodeToCanvas = (node: Konva.Node, bbox?: Rect): HTMLCanvasElement => {
  return node.toCanvas({ ...(bbox ?? {}) });
};

/**
 * Converts a Konva node to a Blob
 * @param node - The Konva node to convert to a Blob
 * @param bbox - The bounding box to crop to
 * @returns A Promise that resolves with Blob of the node cropped to the bounding box
 */
export const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      assert(blob, 'blob is null');
      resolve(blob);
    });
  });
};

export const canvasToImageData = (canvas: HTMLCanvasElement): ImageData => {
  const ctx = canvas.getContext('2d');
  assert(ctx, 'ctx is null');
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
};

/**
 * Converts a Konva node to an ImageData object
 * @param node - The Konva node to convert to an ImageData object
 * @param bbox - The bounding box to crop to
 * @returns A Promise that resolves with ImageData object of the node cropped to the bounding box
 */
export const konvaNodeToImageData = (node: Konva.Node, bbox?: Rect): ImageData => {
  const canvas = konvaNodeToCanvas(node, bbox);
  return canvasToImageData(canvas);
};

/**
 * Converts a Konva node to a Blob
 * @param node - The Konva node to convert to a Blob
 * @param bbox - The bounding box to crop to
 * @returns A Promise that resolves to the Blob or null,
 */
export const konvaNodeToBlob = (node: Konva.Node, bbox?: Rect): Promise<Blob> => {
  const canvas = konvaNodeToCanvas(node, bbox);
  return canvasToBlob(canvas);
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

export function getInpaintMaskLayerClone(arg: { manager: CanvasManager }): Konva.Layer {
  const { manager } = arg;
  const layerClone = manager.inpaintMask.konva.layer.clone();
  const objectGroupClone = manager.inpaintMask.konva.group.clone();

  layerClone.destroyChildren();
  layerClone.add(objectGroupClone);

  objectGroupClone.opacity(1);
  objectGroupClone.cache();

  return layerClone;
}

export function getRegionMaskLayerClone(arg: { manager: CanvasManager; id: string }): Konva.Layer {
  const { id, manager } = arg;

  const canvasRegion = manager.regions.get(id);
  assert(canvasRegion, `Canvas region with id ${id} not found`);

  const layerClone = canvasRegion.konva.layer.clone();
  const objectGroupClone = canvasRegion.konva.group.clone();

  layerClone.destroyChildren();
  layerClone.add(objectGroupClone);

  objectGroupClone.opacity(1);
  objectGroupClone.cache();

  return layerClone;
}

export function getControlAdapterLayerClone(arg: { manager: CanvasManager; id: string }): Konva.Layer {
  const { id, manager } = arg;

  const controlAdapter = manager.controlAdapters.get(id);
  assert(controlAdapter, `Canvas region with id ${id} not found`);

  const controlAdapterClone = controlAdapter.konva.layer.clone();
  const objectGroupClone = controlAdapter.konva.group.clone();

  controlAdapterClone.destroyChildren();
  controlAdapterClone.add(objectGroupClone);

  objectGroupClone.opacity(1);
  objectGroupClone.cache();

  return controlAdapterClone;
}

export function getInitialImageLayerClone(arg: { manager: CanvasManager }): Konva.Layer {
  const { manager } = arg;

  const initialImage = manager.initialImage;

  const initialImageClone = initialImage.konva.layer.clone();
  const objectGroupClone = initialImage.konva.group.clone();

  initialImageClone.destroyChildren();
  initialImageClone.add(objectGroupClone);

  objectGroupClone.opacity(1);
  objectGroupClone.cache();

  return initialImageClone;
}

export function getCompositeLayerStageClone(arg: { manager: CanvasManager }): Konva.Stage {
  const { manager } = arg;

  const layersState = manager.stateApi.getLayersState();

  const stageClone = manager.stage.clone();

  stageClone.scaleX(1);
  stageClone.scaleY(1);
  stageClone.x(0);
  stageClone.y(0);

  const validLayers = layersState.entities.filter(isValidLayer);
  console.log(validLayers);
  // Konva bug (?) - when iterating over the array returned from `stage.getLayers()`, if you destroy a layer, the array
  // is mutated in-place and the next iteration will skip the next layer. To avoid this, we first collect the layers
  // to delete in a separate array and then destroy them.
  // TODO(psyche): Maybe report this?
  const toDelete: Konva.Layer[] = [];

  for (const konvaLayer of stageClone.getLayers()) {
    const layer = validLayers.find((l) => l.id === konvaLayer.id());
    if (!layer) {
      console.log('deleting', konvaLayer);
      toDelete.push(konvaLayer);
    } else {
      konvaLayer.findOne<Konva.Group>(`.${CanvasLayer.GROUP_NAME}`)?.findOne(`.${CanvasLayer.BBOX_NAME}`)?.destroy();
    }
  }

  for (const konvaLayer of toDelete) {
    konvaLayer.destroy();
  }

  return stageClone;
}

export function getGenerationMode(arg: { manager: CanvasManager }): GenerationMode {
  const { manager } = arg;
  const { x, y, width, height } = manager.stateApi.getBbox().rect;
  const inpaintMaskLayer = getInpaintMaskLayerClone(arg);
  const inpaintMaskImageData = konvaNodeToImageData(inpaintMaskLayer, { x, y, width, height });
  const inpaintMaskTransparency = getImageDataTransparency(inpaintMaskImageData);
  const compositeLayer = getCompositeLayerStageClone(arg);
  const compositeLayerImageData = konvaNodeToImageData(compositeLayer, { x, y, width, height });
  imageDataToBlob(compositeLayerImageData).then((blob) => {
    previewBlob(blob, 'composite layer');
  });
  const compositeLayerTransparency = getImageDataTransparency(compositeLayerImageData);
  if (compositeLayerTransparency.isPartiallyTransparent) {
    if (compositeLayerTransparency.isFullyTransparent) {
      return 'txt2img';
    }
    return 'outpaint';
  } else {
    if (!inpaintMaskTransparency.isFullyTransparent) {
      return 'inpaint';
    }
    return 'img2img';
  }
}

export async function getRegionMaskImage(arg: {
  manager: CanvasManager;
  id: string;
  bbox?: Rect;
  preview?: boolean;
}): Promise<ImageDTO> {
  const { manager, id, bbox, preview = false } = arg;
  const region = manager.stateApi.getRegionsState().entities.find((entity) => entity.id === id);
  assert(region, `Region entity state with id ${id} not found`);

  // if (region.imageCache) {
  //   const imageDTO = await this.util.getImageDTO(region.imageCache.name);
  //   if (imageDTO) {
  //     return imageDTO;
  //   }
  // }

  const layerClone = getRegionMaskLayerClone({ id, manager });
  const blob = await konvaNodeToBlob(layerClone, bbox);

  if (preview) {
    previewBlob(blob, `region ${region.id} mask`);
  }

  layerClone.destroy();

  const imageDTO = await manager.util.uploadImage(blob, `${region.id}_mask.png`, 'mask', true);
  manager.stateApi.onRegionMaskImageCached(region.id, imageDTO);
  return imageDTO;
}

export async function getControlAdapterImage(arg: {
  manager: CanvasManager;
  id: string;
  bbox?: Rect;
  preview?: boolean;
}): Promise<ImageDTO> {
  const { manager, id, bbox, preview = false } = arg;
  const ca = manager.stateApi.getControlAdaptersState().entities.find((entity) => entity.id === id);
  assert(ca, `Control adapter entity state with id ${id} not found`);

  // if (region.imageCache) {
  //   const imageDTO = await this.util.getImageDTO(region.imageCache.name);
  //   if (imageDTO) {
  //     return imageDTO;
  //   }
  // }

  const layerClone = getControlAdapterLayerClone({ id, manager });
  const blob = await konvaNodeToBlob(layerClone, bbox);

  if (preview) {
    previewBlob(blob, `region ${ca.id} mask`);
  }

  layerClone.destroy();

  const imageDTO = await manager.util.uploadImage(blob, `${ca.id}_control_image.png`, 'control', true);
  // manager.stateApi.onRegionMaskImageCached(ca.id, imageDTO);
  return imageDTO;
}

export async function getInitialImage(arg: {
  manager: CanvasManager;
  bbox?: Rect;
  preview?: boolean;
}): Promise<ImageDTO> {
  const { manager, bbox, preview = false } = arg;

  // if (region.imageCache) {
  //   const imageDTO = await this.util.getImageDTO(region.imageCache.name);
  //   if (imageDTO) {
  //     return imageDTO;
  //   }
  // }

  const layerClone = getInitialImageLayerClone({ manager });
  const blob = await konvaNodeToBlob(layerClone, bbox);

  if (preview) {
    previewBlob(blob, 'initial image');
  }

  layerClone.destroy();

  const imageDTO = await manager.util.uploadImage(blob, 'initial_image.png', 'other', true);
  // manager.stateApi.onRegionMaskImageCached(ca.id, imageDTO);
  return imageDTO;
}

export async function getInpaintMaskImage(arg: {
  manager: CanvasManager;
  bbox?: Rect;
  preview?: boolean;
}): Promise<ImageDTO> {
  const { manager, bbox, preview = false } = arg;
  // const inpaintMask = this.stateApi.getInpaintMaskState();

  // if (inpaintMask.imageCache) {
  //   const imageDTO = await this.util.getImageDTO(inpaintMask.imageCache.name);
  //   if (imageDTO) {
  //     return imageDTO;
  //   }
  // }

  const layerClone = getInpaintMaskLayerClone({ manager });
  const blob = await konvaNodeToBlob(layerClone, bbox);

  if (preview) {
    previewBlob(blob, 'inpaint mask');
  }

  layerClone.destroy();

  const imageDTO = await manager.util.uploadImage(blob, 'inpaint_mask.png', 'mask', true);
  manager.stateApi.onInpaintMaskImageCached(imageDTO);
  return imageDTO;
}

export async function getCompositeLayerImage(arg: {
  manager: CanvasManager;
  bbox?: Rect;
  preview?: boolean;
}): Promise<ImageDTO> {
  const { manager, bbox, preview = false } = arg;
  // const { imageCache } = this.stateApi.getLayersState();

  // if (imageCache) {
  //   const imageDTO = await this.util.getImageDTO(imageCache.name);
  //   if (imageDTO) {
  //     return imageDTO;
  //   }
  // }

  const stageClone = getCompositeLayerStageClone({ manager });

  const blob = await konvaNodeToBlob(stageClone, bbox);

  if (preview) {
    previewBlob(blob, 'image source');
  }

  stageClone.destroy();

  const imageDTO = await manager.util.uploadImage(blob, 'base_layer.png', 'general', true);
  manager.stateApi.onLayerImageCached(imageDTO);
  return imageDTO;
}

export function loadImage(src: string, imageEl?: HTMLImageElement): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const _imageEl = imageEl ?? new Image();
    _imageEl.onload = () => resolve(_imageEl);
    _imageEl.onerror = (error) => reject(error);
    _imageEl.src = src;
  });
}

/**
 * Generates a random alphanumeric string of length 10. Probably not secure at all.
 */
export const nanoid = customAlphabet('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', 10);

export function getPrefixedId(prefix: string): string {
  return `${prefix}:${nanoid()}`;
}

export function getObjectId(type: RenderableObject['type'], isBuffer?: boolean): string {
  if (isBuffer) {
    return getPrefixedId(`buffer_${type}`);
  } else {
    return getPrefixedId(type);
  }
}

export type Subscription = {
  name: string;
  unsubscribe: () => void;
};

/**
 * Builds a subscribe function for a nanostores atom.
 * @param subscribe The subscribe function of the atom
 * @param name The name of the atom
 * @returns A subscribe function that returns an object with the name and unsubscribe function
 */
export const buildSubscribe = <T>(subscribe: WritableAtom<T>['subscribe'], name: string) => {
  return (cb: Parameters<WritableAtom<T>['subscribe']>[0]): Subscription => {
    return {
      name,
      unsubscribe: subscribe(cb),
    };
  };
};

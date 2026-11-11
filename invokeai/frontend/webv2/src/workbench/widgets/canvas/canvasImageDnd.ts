import type { GalleryImage } from '@features/gallery';
import type { GalleryCanvasImportDestination } from '@workbench/canvas-operations/api';

import { isGalleryImageDragData } from '@features/gallery/utility';
import { findDropZoneAt } from '@platform/browser/dropZones';

export interface CanvasImageDropData {
  kind: 'canvas-image-target';
  destination: Exclude<GalleryCanvasImportDestination, 'regional-guidance'>;
}

export const isCanvasImageDropData = (value: unknown): value is CanvasImageDropData =>
  isRecord(value) && value.kind === 'canvas-image-target' && isCanvasImageDropDestination(value.destination);

export const getCanvasImageDropData = (destination: CanvasImageDropData['destination']): CanvasImageDropData => ({
  destination,
  kind: 'canvas-image-target',
});

export const getCanvasImageDropId = (destination: CanvasImageDropData['destination']): string =>
  `canvas-image-target:${destination}`;

/**
 * Marks each zone's element. A file dragged in from the desktop is an HTML5 drag dnd-kit never sees; it finds its zone
 * the way dnd-kit's collision does, by the zone's rect under the pointer, whatever is painted above it.
 */
export const CANVAS_IMAGE_DROP_DESTINATION_ATTRIBUTE = 'data-canvas-image-drop-destination';

export const findCanvasImageDropDestination = (
  root: Element,
  point: { clientX: number; clientY: number }
): CanvasImageDropData['destination'] | null => {
  const destination = findDropZoneAt(root, CANVAS_IMAGE_DROP_DESTINATION_ATTRIBUTE, point);
  return isCanvasImageDropDestination(destination) ? destination : null;
};

export interface CanvasImageDropResolution {
  destination: CanvasImageDropData['destination'];
  imageNames: string[];
}

export const resolveCanvasImageDrop = (activeData: unknown, overData: unknown): CanvasImageDropResolution | null => {
  if (!isGalleryImageDragData(activeData) || !isCanvasImageDropData(overData)) {
    return null;
  }

  return {
    destination: overData.destination,
    imageNames: activeData.items.map((item) => item.name),
  };
};

export const orderCanvasImageDropImages = (
  imageNames: readonly string[],
  images: readonly GalleryImage[]
): GalleryImage[] => {
  const imagesByName = new Map(images.map((image) => [image.imageName, image]));
  return imageNames.flatMap((imageName) => {
    const image = imagesByName.get(imageName);
    return image ? [image] : [];
  });
};

const canvasImageDropDestinations = {
  control: true,
  'control-resized': true,
  'inpaint-mask': true,
  raster: true,
  'regional-reference': true,
} satisfies Record<CanvasImageDropData['destination'], true>;

const isCanvasImageDropDestination = (value: unknown): value is CanvasImageDropData['destination'] =>
  typeof value === 'string' && Object.hasOwn(canvasImageDropDestinations, value);

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

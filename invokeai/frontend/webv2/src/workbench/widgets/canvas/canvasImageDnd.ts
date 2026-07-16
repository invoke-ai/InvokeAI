import type { GalleryCanvasImportDestination } from '@workbench/canvas-operations/api';
import type { GalleryImage } from '@workbench/gallery/api';

import { isGalleryImageDragData } from '@workbench/widgets/gallery/galleryDnd';

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
    imageNames: activeData.images.map((image) => image.imageName),
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

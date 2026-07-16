import type { GalleryImage } from '@workbench/gallery/api';

import { describe, expect, it } from 'vitest';

import {
  getGalleryCanvasImportMenuItems,
  getImageContextMenuImages,
  getImageContextMenuRecallRequestKey,
} from './ImageContextMenu';

const image: GalleryImage = {
  boardId: 'none',
  height: 512,
  imageCategory: 'general',
  imageName: 'image.png',
  imageUrl: '/image.png',
  queuedAt: '2026-06-15T00:00:00Z',
  sourceQueueItemId: 'queue-item',
  starred: false,
  thumbnailUrl: '/thumb.png',
  width: 512,
};

describe('getImageContextMenuImages', () => {
  it('returns only usable images from a context menu target', () => {
    expect(getImageContextMenuImages({ images: [image], x: 0, y: 0 })).toEqual([image]);
    expect(
      getImageContextMenuImages({ images: [image, { imageName: 'missing-urls.png' } as GalleryImage], x: 0, y: 0 })
    ).toEqual([image]);
  });

  it('tolerates stale or malformed targets', () => {
    expect(getImageContextMenuImages(null)).toEqual([]);
    expect(getImageContextMenuImages({ images: undefined as unknown as GalleryImage[], x: 0, y: 0 })).toEqual([]);
  });
});

describe('getImageContextMenuRecallRequestKey', () => {
  it('uses stable image primitives instead of object identity', () => {
    const refreshedImage = { ...image };

    expect(getImageContextMenuRecallRequestKey(image, false)).toBe(
      getImageContextMenuRecallRequestKey(refreshedImage, false)
    );
  });

  it('does not request single-image recall for empty or bulk menu targets', () => {
    expect(getImageContextMenuRecallRequestKey(null, false)).toBeNull();
    expect(getImageContextMenuRecallRequestKey(image, true)).toBeNull();
  });
});

describe('getGalleryCanvasImportMenuItems', () => {
  const expectedDestinationsAndLabels = [
    ['raster', 'widgets.canvas.import.raster'],
    ['control', 'widgets.canvas.import.control'],
    ['inpaint-mask', 'widgets.canvas.import.inpaintMask'],
    ['regional-guidance', 'widgets.canvas.import.regionalGuidance'],
    ['regional-reference', 'widgets.canvas.import.regionalReference'],
    ['control-resized', 'widgets.canvas.import.resizedControl'],
  ];

  it('returns the six stable single-image canvas import entries in order', () => {
    expect(getGalleryCanvasImportMenuItems(false)).toEqual(
      expectedDestinationsAndLabels.map(([destination, label]) => ({
        destination,
        label,
        value: `send-image-to-canvas-${destination}`,
      }))
    );
  });

  it('returns the same six destinations for bulk imports with stable bulk values', () => {
    expect(getGalleryCanvasImportMenuItems(true)).toEqual(
      expectedDestinationsAndLabels.map(([destination, label]) => ({
        destination,
        label,
        value: `send-images-to-canvas-${destination}`,
      }))
    );
  });
});

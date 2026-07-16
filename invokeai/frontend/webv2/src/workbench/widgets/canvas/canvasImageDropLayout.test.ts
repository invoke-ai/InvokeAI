import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { GalleryImage } from '@workbench/gallery/api';
import type { Project, WorkbenchState } from '@workbench/types';

import { getGalleryImageDragData } from '@workbench/widgets/gallery/galleryDnd';
import { describe, expect, it } from 'vitest';

import {
  getCanvasImageDropData,
  getCanvasImageDropId,
  orderCanvasImageDropImages,
  resolveCanvasImageDrop,
} from './canvasImageDnd';
import { CANVAS_IMAGE_DROP_GRID_LAYOUT, CANVAS_IMAGE_DROP_LAYOUT } from './CanvasImageDropOverlay';
import { executeCanvasImageDropImport } from './executeCanvasImageDropImport';

describe('canvas image drop layout', () => {
  it('covers the full surface with the legacy six-column, two-row grid', () => {
    expect(CANVAS_IMAGE_DROP_GRID_LAYOUT).toEqual({
      gap: '0',
      padding: '0',
      templateColumns: 'repeat(6, minmax(0, 1fr))',
      templateRows: 'repeat(2, minmax(0, 1fr))',
    });
  });

  it('occupies exactly six columns in each of two rows', () => {
    expect(CANVAS_IMAGE_DROP_LAYOUT).toEqual([
      { colSpan: 3, destination: 'raster', row: 1 },
      { colSpan: 3, destination: 'control', row: 1 },
      { colSpan: 2, destination: 'regional-reference', row: 2 },
      { colSpan: 2, destination: 'inpaint-mask', row: 2 },
      { colSpan: 2, destination: 'control-resized', row: 2 },
    ]);

    expect(
      CANVAS_IMAGE_DROP_LAYOUT.reduce<Record<number, number>>((columnsByRow, zone) => {
        columnsByRow[zone.row] = (columnsByRow[zone.row] ?? 0) + zone.colSpan;
        return columnsByRow;
      }, {})
    ).toEqual({ 1: 6, 2: 6 });
  });

  it('has unique destinations and stable unique droppable ids', () => {
    const destinations = CANVAS_IMAGE_DROP_LAYOUT.map((zone) => zone.destination);
    const firstIds = destinations.map(getCanvasImageDropId);
    const secondIds = destinations.map(getCanvasImageDropId);

    expect(new Set(destinations).size).toBe(CANVAS_IMAGE_DROP_LAYOUT.length);
    expect(new Set(firstIds).size).toBe(CANVAS_IMAGE_DROP_LAYOUT.length);
    expect(secondIds).toEqual(firstIds);
  });
});

describe('resolveCanvasImageDrop', () => {
  const galleryDrag = getGalleryImageDragData([
    { boardId: 'board-a', imageName: 'third.png' },
    { boardId: 'board-b', imageName: 'first.png' },
    { boardId: 'board-a', imageName: 'second.png' },
  ]);

  it('accepts a gallery-image to canvas-image-target pair and preserves requested order', () => {
    expect(resolveCanvasImageDrop(galleryDrag, getCanvasImageDropData('control-resized'))).toEqual({
      destination: 'control-resized',
      imageNames: ['third.png', 'first.png', 'second.png'],
    });
  });

  it.each([
    [null, getCanvasImageDropData('raster')],
    [{ kind: 'widget-instance' }, getCanvasImageDropData('raster')],
    [galleryDrag, null],
    [galleryDrag, { destination: 'raster', kind: 'widget-region' }],
    [galleryDrag, { destination: 'regional-guidance', kind: 'canvas-image-target' }],
  ])('rejects a non gallery-image to canvas-image-target pair', (activeData, overData) => {
    expect(resolveCanvasImageDrop(activeData, overData)).toBeNull();
  });
});

describe('orderCanvasImageDropImages', () => {
  it('restores requested order and omits backend results that were not requested', () => {
    const first = { imageName: 'first.png' } as GalleryImage;
    const second = { imageName: 'second.png' } as GalleryImage;
    const extra = { imageName: 'extra.png' } as GalleryImage;

    expect(orderCanvasImageDropImages(['first.png', 'missing.png', 'second.png'], [extra, second, first])).toEqual([
      first,
      second,
    ]);
  });
});

describe('executeCanvasImageDropImport', () => {
  it('looks up all names once, restores requested order, and calls the shared service once', async () => {
    const first = { imageName: 'first.png' } as GalleryImage;
    const second = { imageName: 'second.png' } as GalleryImage;
    const extra = { imageName: 'extra.png' } as GalleryImage;
    const project = { id: 'captured-project' } as Project;
    const engine = { projectId: project.id } as CanvasEngine;
    const getState = () => ({ activeProjectId: project.id }) as WorkbenchState;
    const lookupCalls: string[][] = [];
    const importCalls: unknown[] = [];

    const result = await executeCanvasImageDropImport({
      destination: 'control',
      dispatch: () => undefined,
      engine,
      getGalleryImages: (imageNames) => {
        lookupCalls.push([...imageNames]);
        return Promise.resolve([second, extra, first]);
      },
      getState,
      imageNames: ['first.png', 'second.png'],
      importGalleryImages: (options) => {
        importCalls.push(options);
        return Promise.resolve({ failedImageNames: [], layerIds: ['layer-1', 'layer-2'], status: 'imported' });
      },
      project,
    });

    expect(lookupCalls).toEqual([['first.png', 'second.png']]);
    expect(importCalls).toHaveLength(1);
    expect(importCalls[0]).toMatchObject({
      destination: 'control',
      engine,
      getState,
      images: [first, second],
      project,
    });
    expect(result).toEqual({ failedImageNames: [], layerIds: ['layer-1', 'layer-2'], status: 'imported' });
  });
});

import type { GalleryImage } from '@features/gallery';
import type { sendGalleryImageAsControl } from '@workbench/canvas-operations/api';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchCanvasCommands } from '@workbench/workbenchStore';

import { getGalleryItemDragData } from '@features/gallery/utility';
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
  const galleryDrag = getGalleryItemDragData([
    { kind: 'image', name: 'third.png' },
    { kind: 'image', name: 'first.png' },
    { kind: 'image', name: 'second.png' },
  ]);

  it('accepts an all-image gallery item payload and preserves requested order', () => {
    expect(resolveCanvasImageDrop(galleryDrag, getCanvasImageDropData('control-resized'))).toEqual({
      destination: 'control-resized',
      imageNames: ['third.png', 'first.png', 'second.png'],
    });
  });

  it.each([
    [null, getCanvasImageDropData('raster')],
    [{ kind: 'widget-instance' }, getCanvasImageDropData('raster')],
    [getGalleryItemDragData([{ kind: 'video', name: 'clip.mp4' }]), getCanvasImageDropData('raster')],
    [
      getGalleryItemDragData([
        { kind: 'image', name: 'still.png' },
        { kind: 'video', name: 'clip.mp4' },
      ]),
      getCanvasImageDropData('raster'),
    ],
    [galleryDrag, null],
    [galleryDrag, { destination: 'raster', kind: 'widget-region' }],
    [galleryDrag, { destination: 'regional-guidance', kind: 'canvas-image-target' }],
  ])('rejects non-image-only gallery payloads and invalid canvas drop pairs', (activeData, overData) => {
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
    const queries = {
      getProject: (projectId: string) => (projectId === project.id ? project : null),
      isActiveProject: (projectId: string) => projectId === project.id,
    };
    const lookupCalls: string[][] = [];
    const importCalls: unknown[] = [];

    const result = await executeCanvasImageDropImport({
      canvas: { apply: () => false } as unknown as WorkbenchCanvasCommands,
      destination: 'control-resized',
      engine,
      getGalleryImages: (imageNames) => {
        lookupCalls.push([...imageNames]);
        return Promise.resolve([second, extra, first]);
      },
      queries,
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
      destination: 'control-resized',
      engine,
      getProject: queries.getProject,
      images: [first, second],
      isActiveProject: queries.isActiveProject,
      project,
    });
    expect(result).toEqual({ kind: 'success', options: { count: 2 }, titleKey: 'widgets.canvas.import.success' });
  });

  describe('one image on the control zone', () => {
    const image = { imageName: 'one.png' } as GalleryImage;
    const project = { id: 'captured-project' } as Project;
    const drop = (
      destination: 'control' | 'control-resized',
      sent: Awaited<ReturnType<typeof sendGalleryImageAsControl>>
    ) => {
      const sendCalls: unknown[] = [];
      const importCalls: unknown[] = [];
      const result = executeCanvasImageDropImport({
        canvas: { apply: () => false } as unknown as WorkbenchCanvasCommands,
        destination,
        engine: null,
        getGalleryImages: () => Promise.resolve([image]),
        imageNames: ['one.png'],
        importGalleryImages: (options) => {
          importCalls.push(options);
          return Promise.resolve({ failedImageNames: [], layerIds: ['new-layer'], status: 'imported' });
        },
        project,
        queries: { getProject: () => project, isActiveProject: () => true },
        sendAsControl: (options) => {
          sendCalls.push(options);
          return Promise.resolve(sent);
        },
      });
      return { importCalls, result, sendCalls };
    };

    it('fills the selected empty control layer and names it, as Use as Control does', async () => {
      const { importCalls, result, sendCalls } = drop('control', {
        layerId: 'control-1',
        layerName: 'Depth 1',
        status: 'filled',
      });

      await expect(result).resolves.toEqual({
        kind: 'success',
        options: { name: 'Depth 1' },
        titleKey: 'widgets.canvas.import.controlFilled',
      });
      expect(sendCalls).toEqual([expect.objectContaining({ image, project })]);
      expect(importCalls).toEqual([]);
    });

    it('reports a new layer as an import, and leaves the resized zone to the import', async () => {
      await expect(drop('control', { layerId: 'control-2', status: 'created' }).result).resolves.toEqual({
        kind: 'success',
        options: { count: 1 },
        titleKey: 'widgets.canvas.import.success',
      });

      const resized = drop('control-resized', { layerId: 'unused', status: 'created' });
      await resized.result;
      expect(resized.sendCalls).toEqual([]);
      expect(resized.importCalls).toEqual([expect.objectContaining({ destination: 'control-resized' })]);
    });
  });
});

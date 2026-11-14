/**
 * Selection Fill, Erase, Delete and Cut, and mask transforms, on a real engine in Chromium: the commands the Canvas
 * runs, through the engine's own pixel editor and history.
 */

import type { CanvasEditRefusal, CanvasLayerContract } from '@workbench/canvas-engine/api';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { copyCanvasSelection } from './canvasClipboard';
import { executeCanvasHotkeyCommand } from './canvasHotkeyCommands';

const cleanups: (() => void | Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) {
    await cleanup();
  }
});

/** A 64×64 PNG: opaque blue on its left half, transparent on its right. */
const halfBluePng = (): Promise<Blob> => {
  const canvas = new OffscreenCanvas(64, 64);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#0000ff';
  ctx.fillRect(0, 0, 32, 64);
  return canvas.convertToBlob({ type: 'image/png' });
};

const MASK_BITMAP = { height: 64, imageName: 'mask.png', width: 64 };

/** A real engine over `layers` with `selected` selected, attached to on-page canvases. */
const setup = async (layers: readonly Parameters<typeof stacksFrom>[0][number][], selected: string) => {
  const store = createWorkbenchStore();
  const projectId = store.getState().activeProjectId;
  store.commands.canvas.apply(projectId, {
    document: { ...createEmptyCanvasDocument(256, 256), selectedLayerId: selected, stacks: stacksFrom(layers) },
    type: 'replaceCanvasDocument',
  });
  const png = await halfBluePng();
  const engine = createCanvasEngine({
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.resolve(png),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });
  cleanups.push(() => engine.lifecycle.dispose());
  const [screen, overlay] = [document.createElement('canvas'), document.createElement('canvas')];
  for (const canvas of [screen, overlay]) {
    canvas.width = 256;
    canvas.height = 256;
    canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px';
    document.body.append(canvas);
    cleanups.push(() => canvas.remove());
  }
  engine.surface.attach(screen, overlay);
  engine.interaction.set('clipToBbox', false);
  const refusals: CanvasEditRefusal[] = [];
  engine.tools.onEditRefused((refusal) => refusals.push(refusal));

  const doc = () => store.getState().projects.find((project) => project.id === projectId)!.canvas.document;
  const layer = (id: string) => engine.document.model()!.getLayer(id) as CanvasLayerContract;
  const sourceType = (id: string) => {
    const current = layer(id);
    return current.type === 'raster' ? current.source.type : null;
  };
  const maskBitmap = (id: string) => {
    const current = layer(id);
    return current.type === 'inpaint_mask' ? current.mask.bitmap : null;
  };

  /** Marquee-selects the document rect from (x0, y0) to (x1, y1), replacing the selection. */
  const select = (x0: number, y0: number, x1: number, y1: number) => {
    engine.tools.setTool('marquee');
    const view = engine.viewport.getViewport();
    const box = overlay.getBoundingClientRect();
    const fire = (type: string, x: number, y: number, buttons: number) => {
      const point = view.documentToScreen({ x, y });
      overlay.dispatchEvent(
        new PointerEvent(type, {
          bubbles: true,
          buttons,
          clientX: box.left + point.x,
          clientY: box.top + point.y,
          isPrimary: true,
          pointerId: 1,
        })
      );
    };
    fire('pointerdown', x0, y0, 1);
    fire('pointermove', (x0 + x1) / 2, (y0 + y1) / 2, 1);
    fire('pointermove', x1, y1, 1);
    fire('pointerup', x1, y1, 0);
  };

  /** RGBA of `id`'s pixels at document (x, y), read through its export (layer-local, so placed by its transform). */
  const pixelAt = async (id: string, x: number, y: number): Promise<number[]> => {
    const exported = await engine.exports.exportLayerPixels(id, { includeDisabled: true });
    if (exported.status !== 'ok') {
      return [0, 0, 0, 0];
    }
    try {
      const { transform } = layer(id);
      const lx = Math.floor((x - transform.x) / transform.scaleX) - exported.rect.x;
      const ly = Math.floor((y - transform.y) / transform.scaleY) - exported.rect.y;
      if (lx < 0 || ly < 0 || lx >= exported.rect.width || ly >= exported.rect.height) {
        return [0, 0, 0, 0];
      }
      return [...exported.surface.ctx.getImageData(lx, ly, 1, 1).data];
    } finally {
      exported.release();
    }
  };

  /** Waits until `id`'s pixels have loaded, as an edit needs. */
  const loaded = async (id: string) => {
    await vi.waitFor(async () => {
      expect((await engine.exports.exportLayerPixels(id, { includeDisabled: true })).status).toBe('ok');
    });
  };

  const runCommand = (commandId: string) =>
    executeCanvasHotkeyCommand(commandId, {
      copySelection: vi.fn(),
      cropAllLayers: vi.fn(),
      dispatch: () => true,
      document: doc(),
      engine,
      hasMultipleStagingSlots: false,
      notifyLayerDuplicateFailed: vi.fn(),
      pasteFromClipboard: vi.fn(),
      reportLayerOperation: vi.fn(),
      reportMergeDownRefusal: vi.fn(),
      reportPreparedCommit: vi.fn(),
      reportStructuralCommit: vi.fn(),
      reportTransformRefusal: vi.fn(),
      resetActiveColors: vi.fn(),
      selectedLayerIds: [selected],
      swapActiveColors: vi.fn(),
      t: (key) => key,
    });

  return { doc, engine, layer, loaded, maskBitmap, pixelAt, refusals, runCommand, select, sourceType };
};

const mask = () =>
  layerContract('mask', 'inpaint_mask', {
    mask: { bitmap: MASK_BITMAP, fill: { color: '#e07575', style: 'diagonal' }, offset: { x: 0, y: 0 } },
    name: 'Mask',
  } as Partial<CanvasLayerContract>);

const photo = (overrides: Partial<CanvasLayerContract> = {}) =>
  layerContract('photo', 'raster', {
    name: 'Photo',
    source: { image: { height: 64, imageName: 'photo.png', width: 64 }, type: 'image' },
    ...overrides,
  } as Partial<CanvasLayerContract>);

describe('selection edits on a real engine', () => {
  it('Delete and Cut remove a mask’s selected coverage, one undoable step each', async () => {
    const h = await setup([mask()], 'mask');
    await h.loaded('mask');
    // The mask's image covers its left half (x < 32).
    expect((await h.pixelAt('mask', 10, 10))[3]).toBe(255);

    h.select(0, 0, 16, 64);
    h.runCommand('canvas.deleteSelected');
    expect((await h.pixelAt('mask', 10, 10))[3]).toBe(0);
    expect((await h.pixelAt('mask', 20, 10))[3]).toBe(255);
    expect(h.engine.history.getEntries().past).toContain('Erase selection');

    await h.engine.history.undo();
    expect((await h.pixelAt('mask', 10, 10))[3]).toBe(255);
    await h.engine.history.redo();
    expect((await h.pixelAt('mask', 10, 10))[3]).toBe(0);

    const written: Blob[] = [];
    h.select(16, 0, 24, 64);
    await copyCanvasSelection(h.engine.selection, true, (blob) => {
      written.push(blob);
      return Promise.resolve();
    });
    expect(written).toHaveLength(1);
    expect((await h.pixelAt('mask', 20, 10))[3]).toBe(0);
    expect((await h.pixelAt('mask', 28, 10))[3]).toBe(255);
    expect(h.refusals).toEqual([]);
  });

  it('a Cut the target refuses writes nothing and says why; Delete says the same', async () => {
    const h = await setup([groupContract('group', [layerContract('inner', 'raster', { name: 'Inner' })])], 'group');
    h.select(0, 0, 32, 32);
    const write = vi.fn(() => Promise.resolve());

    await copyCanvasSelection(h.engine.selection, true, write);
    h.runCommand('canvas.deleteSelected');

    expect(write).not.toHaveBeenCalled();
    const refusal = { layerId: 'group', layerName: 'group', status: 'selection-erase-unsupported' };
    expect(h.refusals).toEqual([refusal, refusal]);
    // Only the marquee's own step: nothing was erased.
    expect(h.engine.history.getEntries().past).toEqual(['Select']);
  });

  it('fills an image in place as one step that undo and redo replay', async () => {
    const h = await setup([photo()], 'photo');
    await h.loaded('photo');
    h.engine.interaction.set('brushOptions', { ...h.engine.interaction.get('brushOptions'), color: '#ff0000' });

    h.select(40, 0, 56, 16);
    h.engine.selection.fillSelection();
    expect(h.sourceType('photo')).toBe('paint');
    expect(await h.pixelAt('photo', 48, 8)).toEqual([255, 0, 0, 255]);
    expect(await h.pixelAt('photo', 8, 8)).toEqual([0, 0, 255, 255]);

    await h.engine.history.undo();
    expect(h.sourceType('photo')).toBe('image');
    await h.loaded('photo');
    expect((await h.pixelAt('photo', 48, 8))[3]).toBe(0);
    await h.engine.history.redo();
    expect(await h.pixelAt('photo', 48, 8)).toEqual([255, 0, 0, 255]);
  });

  it('fills only the existing pixels of a transparency-locked image', async () => {
    const h = await setup([photo({ isTransparencyLocked: true } as Partial<CanvasLayerContract>)], 'photo');
    await h.loaded('photo');
    h.engine.interaction.set('brushOptions', { ...h.engine.interaction.get('brushOptions'), color: '#ff0000' });

    h.select(16, 0, 48, 16);
    h.engine.selection.fillSelection();

    expect(await h.pixelAt('photo', 24, 8)).toEqual([255, 0, 0, 255]);
    expect((await h.pixelAt('photo', 40, 8))[3]).toBe(0);
  });

  it('applies a mask transform as one step that undo and redo replay', async () => {
    const h = await setup([mask()], 'mask');
    await h.loaded('mask');

    expect(h.engine.layers.beginTransform('mask')).toBe('started');
    h.engine.layers.updateTransformSession({ rotation: 0, scaleX: 2, scaleY: 1, x: 30, y: 0 });
    h.engine.layers.applyTransform();
    const moved = { rotation: 0, scaleX: 2, scaleY: 1, x: 30, y: 0 };
    expect(h.layer('mask').transform).toEqual(moved);
    // Not baked: the mask keeps its own image.
    expect(h.maskBitmap('mask')).toEqual(MASK_BITMAP);

    await h.engine.history.undo();
    expect(h.layer('mask').transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
    await h.engine.history.redo();
    expect(h.layer('mask').transform).toEqual(moved);
  });

  it('refuses Delete while a transform session previews another placement', async () => {
    const h = await setup([mask()], 'mask');
    await h.loaded('mask');
    h.select(0, 0, 16, 64);
    expect(h.engine.layers.beginTransform('mask')).toBe('started');
    h.engine.layers.updateTransformSession({ rotation: 0, scaleX: 1, scaleY: 1, x: 100, y: 0 });

    h.runCommand('canvas.deleteSelected');

    expect(h.refusals).toEqual([{ layerId: 'mask', layerName: 'Mask', status: 'transform-session' }]);
    expect((await h.pixelAt('mask', 10, 10))[3]).toBe(255);
  });
});

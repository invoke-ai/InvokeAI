import type {
  CanvasStackForests,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { PlacedSurface, Rect } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { describe, expect, it } from 'vitest';

import { REPLAY_RASTER_REFUSAL } from './editSteps';
import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { NewRasterLayerController } from './newRasterLayerController';

const paintLayer = (id: string, transform: Partial<CanvasLayerContract['transform']> = {}): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0, ...transform },
  type: 'raster',
});

const makeDoc = (stacks: CanvasStackForests, selectedLayerId: string | null): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks,
  selectedLayerId,
  version: 3,
  width: 100,
});

/** The created layer, narrowed to the raster variant the controller always makes. */
const createdRaster = (document: CanvasDocumentContractV3 | null, id: string): CanvasRasterLayerContractV2 => {
  const layer = getDocumentLeaves(document).find((candidate) => candidate.id === id);
  if (!layer || layer.type !== 'raster') {
    throw new Error(`expected a raster layer ${id}`);
  }
  return layer;
};

const imageData = (width: number, height: number): ImageData =>
  ({ colorSpace: 'srgb', data: new Uint8ClampedArray(width * height * 4), height, width }) as ImageData;

const RECT: Rect = { height: 10, width: 10, x: 5, y: 5 };

interface HarnessOptions {
  stacks?: CanvasStackForests;
  selectedLayerId?: string | null;
  maskRect?: Rect | null;
  cacheRect?: Rect;
  gestureActive?: boolean;
  historyBytes?: number;
}

const createHarness = (options: HarnessOptions = {}) => {
  const stacks = options.stacks ?? stacksFrom([paintLayer('a')]);
  const selectedLayerId =
    options.selectedLayerId === undefined ? (stacks.raster[0]?.id ?? null) : options.selectedLayerId;
  const h = createLayerOperationHarness(makeDoc(stacks, selectedLayerId), options);
  const maskRect = options.maskRect === undefined ? { height: 30, width: 30, x: 20, y: 20 } : options.maskRect;
  const mask: PlacedSurface | null = maskRect
    ? { rect: maskRect, surface: h.backend.createSurface(maskRect.width, maskRect.height) }
    : null;
  if (selectedLayerId) {
    h.layers.getOrCreateRect(selectedLayerId, options.cacheRect ?? { height: 100, width: 100, x: 0, y: 0 });
  }
  const controller = new NewRasterLayerController({
    backend: h.backend,
    ctx: h.ctx,
    layers: h.layers,
    selection: { mask: () => mask } as SelectionState,
  });
  const insert = () => controller.insert(RECT, h.backend.createSurface(10, 10), 'New', 'Insert');
  return { ...h, controller, insert };
};

const leafIds = (document: CanvasDocumentContractV3 | null): string[] =>
  getDocumentLeaves(document).map((layer) => layer.id);

describe('NewRasterLayerController: insert', () => {
  it('inserts a paint layer above the active one, selects it and installs its pixels', () => {
    const h = createHarness({
      stacks: stacksFrom([paintLayer('top'), paintLayer('bottom')]),
      selectedLayerId: 'bottom',
    });

    expect(h.insert()).toEqual({ layerId: 'new-1', status: 'created' });
    // Index 0 is top-most, so "above bottom" is bottom's own index.
    expect(leafIds(h.document())).toEqual(['top', 'new-1', 'bottom']);
    expect(h.document().selectedLayerId).toBe('new-1');
    expect(h.layers.get('new-1')?.rect).toEqual(RECT);
    expect(h.history.entries().past).toEqual(['Insert']);
  });

  it('places the pixels via the source offset, leaving the transform identity', () => {
    const h = createHarness();
    h.controller.insert({ height: 10, width: 10, x: 40, y: 60 }, h.backend.createSurface(10, 10), 'New', 'Insert');

    const created = createdRaster(h.document(), 'new-1');
    expect(created.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
    expect(created.source).toMatchObject({ offset: { x: 40, y: 60 }, type: 'paint' });
  });

  it('undo removes the layer and restores the previous selection; redo reinstalls its pixels', async () => {
    const h = createHarness();
    h.insert();

    await expect(h.history.undo()).resolves.toEqual({ status: 'applied' });
    expect(leafIds(h.document())).toEqual(['a']);
    expect(h.document().selectedLayerId).toBe('a');

    h.layers.delete('new-1');
    await expect(h.history.redo()).resolves.toEqual({ status: 'applied' });
    expect(createdRaster(h.document(), 'new-1')).toBeDefined();
    expect(h.document().selectedLayerId).toBe('new-1');
    expect(h.layers.get('new-1')?.rect).toEqual(RECT);
  });

  it('refuses an edit it could never undo before anything changes', () => {
    const h = createHarness({ historyBytes: 100 });
    const before = h.document();

    expect(h.insert()).toEqual({ status: 'over-budget' });
    expect(h.document()).toBe(before);
    expect(h.dispatched).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
  });

  it('refuses when its pixels cannot be reserved, leaving document and history untouched', () => {
    const h = createHarness();
    h.state.rasterBytes = 100;
    const before = h.document();

    expect(h.insert()).toEqual({ status: 'over-budget' });
    expect(h.document()).toBe(before);
    expect(h.installed).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
    expect(h.reservedBytes()).toBe(0);
  });

  it('keeps the step on the redo stack when its replay cannot reserve memory', async () => {
    const h = createHarness();
    h.insert();
    await h.history.undo();
    h.state.rasterBytes = 0;

    const result = await h.history.redo();
    expect(result).toMatchObject({ error: new Error(REPLAY_RASTER_REFUSAL), status: 'failed' });
    expect(leafIds(h.document())).toEqual(['a']);
    expect(h.history.canRedo()).toBe(true);
    expect(h.reservedBytes()).toBe(0);
  });

  it('reports a reducer refusal as failed and records nothing', () => {
    const h = createHarness();
    h.state.refuse = () => true;

    expect(h.insert()).toEqual({ status: 'failed' });
    expect(h.installed).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
    expect(h.reservedBytes()).toBe(0);
  });

  it('rolls back an insertion whose postconditions fail and reports it as failed', () => {
    const h = createHarness();
    const before = leafIds(h.document());
    h.state.interleave = (mutation) =>
      mutation.type === 'applyCanvasLayerStackMutation' && mutation.add
        ? { id: 'a', type: 'setCanvasSelectedLayer' }
        : null;

    expect(h.insert()).toEqual({ status: 'failed' });
    expect(leafIds(h.document())).toEqual(before);
    expect(h.document().selectedLayerId).toBe('a');
    expect(h.installed).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
    expect(h.reservedBytes()).toBe(0);
  });

  it('refuses an empty rect', () => {
    const h = createHarness();
    expect(
      h.controller.insert({ height: 0, width: 0, x: 0, y: 0 }, h.backend.createSurface(0, 0), 'New', 'Insert')
    ).toEqual({ status: 'empty' });
  });

  it('refuses mid-gesture', () => {
    expect(createHarness({ gestureActive: true }).insert()).toEqual({ status: 'busy' });
  });

  it('refuses once disposed', () => {
    const h = createHarness();
    h.controller.dispose();
    expect(h.insert()).toEqual({ status: 'not-ready' });
  });
});

describe('NewRasterLayerController: pasteImage', () => {
  it('centres the pasted pixels on the given point', () => {
    const h = createHarness();
    h.controller.pasteImage(imageData(20, 10), 'Pasted', 'Paste', { x: 50, y: 50 });
    expect(createdRaster(h.document(), 'new-1').source).toMatchObject({ offset: { x: 40, y: 45 } });
  });

  it('places at the origin with no centre given', () => {
    const h = createHarness();
    h.controller.pasteImage(imageData(20, 10), 'Pasted', 'Paste');
    expect(createdRaster(h.document(), 'new-1').source).toMatchObject({ offset: { x: 0, y: 0 } });
  });

  it('refuses zero-sized pixels', () => {
    const h = createHarness();
    expect(h.controller.pasteImage(imageData(0, 0), 'Pasted', 'Paste')).toEqual({ status: 'empty' });
  });
});

describe('NewRasterLayerController: liftSelectionToLayer', () => {
  it('creates a layer covering the selection ∩ content region', () => {
    const h = createHarness();
    expect(h.controller.liftSelectionToLayer('Selection', 'Layer via copy')).toEqual({
      layerId: 'new-1',
      status: 'created',
    });
    expect(createdRaster(h.document(), 'new-1').source).toMatchObject({ offset: { x: 20, y: 20 } });
  });

  it('leaves the source layer untouched — it is a copy, not a cut', () => {
    const h = createHarness();
    const before = h.layers.get('a')!.version;
    h.controller.liftSelectionToLayer('Selection', 'Layer via copy');
    expect(h.layers.get('a')!.version).toBe(before);
  });

  it('bakes the source layer transform into the new layer placement', () => {
    // The copy comes out layer-local; a new layer lives in document space, so a
    // translated source must shift where the pixels land.
    const h = createHarness({ stacks: stacksFrom([paintLayer('a', { x: 10, y: 5 })]) });
    h.controller.liftSelectionToLayer('Selection', 'Layer via copy');

    const created = createdRaster(h.document(), 'new-1');
    expect(created.source).toMatchObject({ offset: { x: 20, y: 20 } });
    expect(created.transform).toMatchObject({ x: 0, y: 0 });
  });

  it('refuses with no selection', () => {
    const h = createHarness({ maskRect: null });
    expect(h.controller.liftSelectionToLayer('Selection', 'Layer via copy')).toEqual({ status: 'empty' });
  });

  it('refuses with no selected layer', () => {
    const h = createHarness({ selectedLayerId: null });
    expect(h.controller.liftSelectionToLayer('Selection', 'Layer via copy')).toEqual({ status: 'empty' });
  });

  it('refuses when the selection misses the layer content', () => {
    const h = createHarness({ cacheRect: { height: 5, width: 5, x: 0, y: 0 } });
    expect(h.controller.liftSelectionToLayer('Selection', 'Layer via copy')).toEqual({ status: 'empty' });
  });
});

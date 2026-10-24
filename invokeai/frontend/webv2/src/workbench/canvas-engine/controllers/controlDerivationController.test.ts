import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasLayerContract,
} from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { describe, expect, it, vi } from 'vitest';

import { ControlDerivationController } from './controlDerivationController';
import { createLayerOperationHarness } from './layerOperationHarness.testing';

const DEPTH = { settings: { model_size: 'small' }, type: 'depth_anything_depth_estimation' };
const PHOTO = { height: 64, imageName: 'photo', width: 64 };
const MAP = { height: 32, imageName: 'depth-map', width: 32 };
const MAP_RECT: Rect = { height: 32, width: 32, x: 4, y: 8 };
const CACHE_RECT: Rect = { height: 20, width: 20, x: 0, y: 0 };

const controlLayer = (overrides: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  id: 'control',
  isEnabled: true,
  isLocked: false,
  name: 'Control Layer 1',
  opacity: 1,
  source: { image: PHOTO, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
  ...overrides,
});

const rasterLayer: CanvasLayerContract = {
  blendMode: 'normal',
  id: 'raster',
  isEnabled: true,
  isLocked: false,
  name: 'Raster',
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
};

const createHarness = (
  layer: CanvasControlLayerContract = controlLayer(),
  options: { historyBytes?: number; pendingPixels?: boolean; lockedParent?: boolean } = {}
) => {
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: { height: 64, width: 64, x: 0, y: 0 },
    height: 64,
    selectedLayerId: layer.id,
    stacks: stacksFrom([
      rasterLayer,
      options.lockedParent ? groupContract('group', [layer], { isLocked: true }) : layer,
    ]),
    version: 3,
    width: 64,
  };
  const endStructuralPreview = vi.fn();
  const h = createLayerOperationHarness(document, { endStructuralPreview, historyBytes: options.historyBytes });
  h.layers.getOrCreateRect(layer.id, CACHE_RECT);
  const fake = { documentRevision: 1, pendingPixels: options.pendingPixels ?? false, sessionHeld: false };
  const discardPersisted = vi.fn();
  const decodeImage = vi.fn((image: { width: number; height: number }) =>
    Promise.resolve({
      decodedHeight: image.height,
      decodedWidth: image.width,
      status: 'ok' as const,
      surface: h.backend.createSurface(image.width, image.height),
    })
  );
  const controller = new ControlDerivationController({
    captureCache: (target) => {
      const entry = h.layers.get(target.id);
      if (!entry) {
        return null;
      }
      const pixels = h.backend.createSurface(entry.rect.width, entry.rect.height);
      pixels.ctx.drawImage(entry.surface.canvas, 0, 0);
      return { pixels, rect: { ...entry.rect } };
    },
    ctx: h.ctx,
    decodeImage,
    discardPersisted,
    getDocumentRevision: () => fake.documentRevision,
    hasPendingPixelWork: () => fake.pendingPixels,
    isLayerSessionHeld: () => fake.sessionHeld,
  });
  const live = (): CanvasControlLayerContract => getDocumentLayer(h.document(), layer.id) as CanvasControlLayerContract;
  /** Commits `patch` against the live layer, as the coordinator's intents do. */
  const patch = (label: string, change: Partial<CanvasControlLayerContract>, coalesceKey?: string) =>
    controller.commitControlLayerPatch({ coalesceKey, expected: live(), label, next: { ...live(), ...change } });
  const commitMap = (guard = controller.captureControlDetectionGuard(layer.id)!) =>
    controller.commitControlMap({ detection: DEPTH, guard, image: MAP, rect: MAP_RECT });
  return { ...h, commitMap, controller, decodeImage, discardPersisted, endStructuralPreview, fake, live, patch };
};

describe('ControlDerivationController.commitControlLayerPatch', () => {
  it('commits one user entry whose undo and redo restore the exact contracts', async () => {
    const h = createHarness();
    const before = h.live();

    expect(h.patch('Set control type', { controlType: 'depth', detection: DEPTH })).toEqual({ status: 'committed' });
    const after = h.live();
    expect(after).toMatchObject({ controlType: 'depth', detection: DEPTH });
    expect(h.history.entries().past).toEqual(['Set control type']);
    expect(h.committedEdits).toHaveLength(1);

    await h.history.undo();
    expect(h.live()).toEqual(before);
    await h.history.redo();
    expect(h.live()).toEqual(after);
  });

  it('coalesces consecutive patches with the same key into the top entry', async () => {
    const h = createHarness(controlLayer({ controlType: 'depth', detection: DEPTH }));
    const before = h.live();

    h.patch('Detection settings', { detection: { ...DEPTH, settings: { model_size: 'base' } } }, 'settings');
    h.patch('Detection settings', { detection: { ...DEPTH, settings: { model_size: 'large' } } }, 'settings');
    expect(h.history.entries().past).toEqual(['Detection settings']);

    h.patch('Detection settings', { detection: DEPTH }, 'other');
    expect(h.history.entries().past).toHaveLength(2);

    await h.history.undo();
    await h.history.undo();
    expect(h.live()).toEqual(before);
  });

  it('refuses a patch prepared against a layer that has changed since', () => {
    const h = createHarness();
    const expected = h.live();
    h.patch('Weight', { adapter: { ...expected.adapter, weight: 0.5 } });

    expect(
      h.controller.commitControlLayerPatch({ expected, label: 'Set type', next: { ...expected, controlType: 'pose' } })
    ).toEqual({ status: 'stale' });
    expect(h.history.entries().past).toEqual(['Weight']);
  });

  it('keeps the outgoing unpersisted pixels when it replaces the source', async () => {
    const painted = controlLayer({ source: { bitmap: null, type: 'paint' } });
    const h = createHarness(painted, { pendingPixels: true });

    h.patch('Replace image', { source: { image: PHOTO, type: 'image' } });
    expect(h.discardPersisted).toHaveBeenCalledWith('control');
    const installedBeforeUndo = h.installed.length;

    await h.history.undo();
    expect(h.live()).toEqual(painted);
    expect(h.installed).toHaveLength(installedBeforeUndo + 1);
    expect(h.layers.get('control')?.rect).toEqual(CACHE_RECT);
  });

  it('leaves pending persistence alone when the source does not change', () => {
    const h = createHarness(controlLayer(), { pendingPixels: true });

    h.patch('Set control type', { controlType: 'depth', detection: DEPTH });
    expect(h.discardPersisted).not.toHaveBeenCalled();
  });

  it('is refused while an operation holds the edit lease or a session holds the layer', () => {
    const h = createHarness();
    h.setEditingLocked(true);
    expect(h.patch('Set control type', { controlType: 'depth' })).toEqual({ status: 'busy' });
    h.setEditingLocked(false);
    h.fake.sessionHeld = true;
    expect(h.patch('Set control type', { controlType: 'depth' })).toEqual({ status: 'busy' });
    expect(h.history.entries().past).toEqual([]);
    expect(h.dispatched).toEqual([]);
  });

  it('is refused when its entry could never be kept', () => {
    const h = createHarness(controlLayer(), { historyBytes: 16 });

    expect(h.patch('Set control type', { controlType: 'depth' })).toEqual({ status: 'over-budget' });
    expect(h.dispatched).toEqual([]);
  });
});

describe('ControlDerivationController.commitControlMap', () => {
  it('lands in the intent entry, so one undo restores the pre-intent layer', async () => {
    const h = createHarness();
    const untyped = h.live();
    h.patch('Set control type', { controlType: 'depth', detection: DEPTH });

    await expect(h.commitMap()).resolves.toEqual({ status: 'committed' });
    const mapped = h.live();
    expect(mapped).toMatchObject({
      detection: DEPTH,
      filter: DEPTH,
      original: { image: PHOTO, type: 'image' },
      source: { bitmap: MAP, offset: { x: 4, y: 8 }, type: 'paint' },
    });
    expect(h.history.entries().past).toEqual(['Set control type']);
    expect(h.layers.get('control')?.rect).toEqual(MAP_RECT);
    expect(h.discardPersisted).toHaveBeenCalledWith('control');
    // The patch routed as a user edit; the background map did not.
    expect(h.committedEdits).toHaveLength(1);

    await h.history.undo();
    expect(h.live()).toEqual(untyped);
    await h.history.redo();
    expect(h.live()).toEqual(mapped);
    expect(h.layers.get('control')?.rect).toEqual(MAP_RECT);
  });

  it('restores the unpersisted map paint the intent and its map replaced', async () => {
    const painted = controlLayer({
      controlType: 'depth',
      detection: DEPTH,
      filter: DEPTH,
      original: { image: PHOTO, type: 'image' },
      source: { bitmap: { height: 20, imageName: 'old-map', width: 20 }, type: 'paint' },
    });
    const h = createHarness(painted, { pendingPixels: true });
    h.patch('Detection settings', { detection: { ...DEPTH, settings: { model_size: 'base' } } }, 'settings');

    await expect(h.commitMap()).resolves.toEqual({ status: 'committed' });
    expect(h.layers.get('control')?.rect).toEqual(MAP_RECT);

    await h.history.undo();
    expect(h.live()).toEqual(painted);
    expect(h.layers.get('control')?.rect).toEqual(CACHE_RECT);
  });

  it('records its own entry when the intent is no longer the newest step', async () => {
    const h = createHarness();
    h.patch('Set control type', { controlType: 'depth', detection: DEPTH });
    const intent = h.live();
    const guard = h.controller.captureControlDetectionGuard('control')!;
    h.patch('Weight', { adapter: { ...intent.adapter, weight: 0.4 } });

    await expect(h.commitMap(guard)).resolves.toEqual({ status: 'committed' });
    expect(h.history.entries().past).toEqual(['Set control type', 'Weight', 'Detect depth']);

    await h.history.undo();
    expect(h.live()).toEqual({ ...intent, adapter: { ...intent.adapter, weight: 0.4 } });
  });

  it('is not staled by weight, name or painting on the map', async () => {
    const h = createHarness(
      controlLayer({
        controlType: 'depth',
        detection: DEPTH,
        original: { image: PHOTO, type: 'image' },
        source: { bitmap: { height: 20, imageName: 'old-map', width: 20 }, type: 'paint' },
      })
    );
    const guard = h.controller.captureControlDetectionGuard('control')!;
    h.patch('Rename', { adapter: { ...h.live().adapter, weight: 0.2 }, name: 'Depth 1' });
    h.ctx.dispatch({
      id: 'control',
      source: { bitmap: { height: 20, imageName: 'painted-map', width: 20 }, type: 'paint' },
      type: 'updateCanvasLayerSource',
    });

    await expect(h.commitMap(guard)).resolves.toEqual({ status: 'committed' });
    expect(h.live()).toMatchObject({ adapter: { weight: 0.2 }, name: 'Depth 1', source: { bitmap: MAP } });
  });

  it.each([
    ['the document was replaced', (h: ReturnType<typeof createHarness>) => (h.fake.documentRevision += 1)],
    [
      'the detection input changed',
      (h: ReturnType<typeof createHarness>) =>
        h.patch('Detection settings', { detection: { ...DEPTH, settings: { model_size: 'large' } } }),
    ],
  ])('records nothing when %s', async (_case, change) => {
    const h = createHarness(controlLayer({ controlType: 'depth', detection: DEPTH }));
    const guard = h.controller.captureControlDetectionGuard('control')!;
    change(h);
    const document = h.document();
    const entries = h.history.entries();

    await expect(h.commitMap(guard)).resolves.toEqual({ status: 'stale' });
    expect(h.document()).toBe(document);
    expect(h.history.entries()).toEqual(entries);
  });

  it('is refused while an operation holds the edit lease or a session holds the layer', async () => {
    const h = createHarness(controlLayer({ controlType: 'depth', detection: DEPTH }));
    const guard = h.controller.captureControlDetectionGuard('control')!;

    h.setEditingLocked(true);
    await expect(h.commitMap(guard)).resolves.toEqual({ status: 'busy' });
    h.setEditingLocked(false);
    h.fake.sessionHeld = true;
    await expect(h.commitMap(guard)).resolves.toEqual({ status: 'busy' });
    expect(h.dispatched).toEqual([]);

    h.fake.sessionHeld = false;
    await expect(h.commitMap(guard)).resolves.toEqual({ status: 'committed' });
  });

  it('waits for unpersisted paint on its own input to reach the original it records', async () => {
    const drawn = controlLayer({
      controlType: 'depth',
      detection: DEPTH,
      source: { bitmap: { height: 20, imageName: 'drawn', width: 20 }, type: 'paint' },
    });
    const h = createHarness(drawn, { pendingPixels: true });

    await expect(h.commitMap()).resolves.toEqual({ status: 'busy' });
    expect(h.dispatched).toEqual([]);
  });

  it('refuses maps it cannot represent and pixels it cannot prepare', async () => {
    const h = createHarness();
    await expect(h.commitMap()).resolves.toEqual({ status: 'unrepresentable' });

    const typed = createHarness(controlLayer({ controlType: 'depth', detection: DEPTH }));
    typed.state.rasterBytes = 0;
    await expect(typed.commitMap()).resolves.toEqual({ status: 'over-budget' });
    expect(typed.dispatched).toEqual([]);
    expect(typed.history.entries().past).toEqual([]);
  });

  it('does not decode for a caller that already aborted', async () => {
    const h = createHarness(controlLayer({ controlType: 'depth', detection: DEPTH }));
    const guard = h.controller.captureControlDetectionGuard('control')!;

    await expect(
      h.controller.commitControlMap({
        detection: DEPTH,
        guard,
        image: MAP,
        rect: MAP_RECT,
        signal: AbortSignal.abort(),
      })
    ).resolves.toEqual({ status: 'aborted' });
    expect(h.decodeImage).not.toHaveBeenCalled();
  });
});

describe('ControlDerivationController refusals', () => {
  const typed = () => controlLayer({ controlType: 'depth', detection: DEPTH });

  it('refuses both commits under a locked ancestor, before decoding', async () => {
    const h = createHarness(typed(), { lockedParent: true });
    const guard = h.controller.captureControlDetectionGuard('control')!;

    expect(h.patch('Set control type', { controlType: 'pose' })).toEqual({ status: 'locked' });
    await expect(h.commitMap(guard)).resolves.toEqual({ status: 'locked' });
    expect(h.decodeImage).not.toHaveBeenCalled();
    expect(h.dispatched).toEqual([]);
  });

  it('reports a removed layer as missing', async () => {
    const h = createHarness(typed());
    const expected = h.live();
    const guard = h.controller.captureControlDetectionGuard('control')!;
    h.ctx.dispatch({
      enabledUpdates: [],
      removeIds: ['control'],
      selectedLayerId: null,
      type: 'applyCanvasLayerStackMutation',
    });

    expect(
      h.controller.commitControlLayerPatch({ expected, label: 'Set type', next: { ...expected, controlType: 'pose' } })
    ).toEqual({ status: 'missing' });
    await expect(h.commitMap(guard)).resolves.toEqual({ status: 'missing' });
    expect(h.history.entries().past).toEqual([]);
  });

  it('retries a decode the edit lock interrupted as busy', async () => {
    const h = createHarness(typed());
    h.decodeImage.mockResolvedValueOnce({ status: 'stale' } as never);

    await expect(h.commitMap()).resolves.toEqual({ status: 'busy' });
    expect(h.dispatched).toEqual([]);
  });

  it('leaves an open structural preview alone until a commit is admitted', async () => {
    const h = createHarness(typed());
    const guard = h.controller.captureControlDetectionGuard('control')!;
    h.fake.sessionHeld = true;
    await expect(h.commitMap(guard)).resolves.toEqual({ status: 'busy' });
    expect(h.endStructuralPreview).not.toHaveBeenCalled();

    h.fake.sessionHeld = false;
    await expect(h.commitMap(guard)).resolves.toEqual({ status: 'committed' });
    expect(h.endStructuralPreview).toHaveBeenCalled();
  });
});

describe('ControlDerivationController.captureControlDetectionGuard', () => {
  it('captures only control layers', () => {
    const h = createHarness();

    expect(h.controller.captureControlDetectionGuard('control')).toMatchObject({
      documentRevision: 1,
      layerId: 'control',
      projectId: h.ctx.projectId,
    });
    expect(h.controller.captureControlDetectionGuard('raster')).toBeNull();
    expect(h.controller.captureControlDetectionGuard('absent')).toBeNull();
  });
});

import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { SamPreviewState } from '@workbench/canvas-engine/controllers/previewStateController';
import type { FloatingSelection } from '@workbench/canvas-engine/selection/floatingSelection';
import type { Mat2d, ToolId } from '@workbench/canvas-engine/types';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { FloatingSelectionFrame } from './floatingSelectionFrame';

import { createOverlayFrame, type CreateOverlayFrameDeps } from './overlayFrame';

const VIEW = [1, 0, 0, 1, 0, 0] as unknown as Mat2d;

const layer = (id: string, overrides: Record<string, unknown> = {}) => ({
  id,
  isEnabled: true,
  name: id,
  source: { image: { height: 8, imageName: `${id}.png`, width: 8 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const documentOf = (layerIds: readonly string[] = ['a'], overrides: Record<string, unknown> = {}) =>
  ({
    bbox: { height: 32, width: 32, x: 0, y: 0 },
    height: 64,
    layers: layerIds.map((id) => layer(id)),
    selectedLayerId: layerIds[0] ?? null,
    width: 64,
    ...overrides,
  }) as unknown as CanvasDocumentContractV2;

interface Harness {
  deps: CreateOverlayFrameDeps;
  stores: ReturnType<typeof createEngineStores>;
  overrides: Map<string, { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }>;
  state: { tool: ToolId; cursor: { point: { x: number; y: number }; radiusDoc: number } | null; phase: number };
  float: { value: FloatingSelection | null };
  selection: { hasSelection: ReturnType<typeof vi.fn>; antsPaths: ReturnType<typeof vi.fn> };
}

let harness: Harness;

const makeHarness = (): Harness => {
  const stores = createEngineStores();
  const overrides = new Map<string, { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }>();
  const state: Harness['state'] = { cursor: null, phase: 0, tool: 'view' };
  const float: Harness['float'] = { value: null };
  const selection = { antsPaths: vi.fn(() => []), hasSelection: vi.fn(() => false) };
  return {
    deps: {
      getActiveToolId: () => state.tool,
      getAntsPhase: () => state.phase,
      getFloatingSelection: () => float.value,
      getOverlayCursor: () => state.cursor,
      selection: selection as unknown as CreateOverlayFrameDeps['selection'],
      stores,
      transformOverrides: overrides,
    },
    float,
    overrides,
    selection,
    state,
    stores,
  };
};

const describeOverlay = (doc = documentOf(), floatFrame: FloatingSelectionFrame | null = null, sam = null) =>
  createOverlayFrame(harness.deps).describe(doc, VIEW, floatFrame, sam);

beforeEach(() => {
  harness = makeHarness();
});

describe('the bbox', () => {
  it('shows the committed frame when nothing is being dragged', () => {
    expect(describeOverlay().bbox).toEqual({ height: 32, width: 32, x: 0, y: 0 });
  });

  it('lets a live drag preview stand in for the committed frame', () => {
    harness.stores.bboxPreview.set({ height: 4, width: 4, x: 9, y: 9 });
    expect(describeOverlay().bbox).toEqual({ height: 4, width: 4, x: 9, y: 9 });
  });

  it('draws handles only while the bbox tool is active', () => {
    expect(describeOverlay().bboxHandles).toBe(false);
    harness.state.tool = 'bbox';
    expect(describeOverlay().bboxHandles).toBe(true);
  });

  it('forces the frame visible under the bbox tool even with the setting off', () => {
    harness.stores.showBbox.set(false);
    expect(describeOverlay().showBbox).toBe(false);
    harness.state.tool = 'bbox';
    // Without a frame the handles would have nothing to attach to.
    expect(describeOverlay().showBbox).toBe(true);
  });
});

describe('the move outline', () => {
  it('is absent for every tool but move', () => {
    expect(describeOverlay().layerOutline).toBeNull();
  });

  it('outlines the selected layer under the move tool', () => {
    harness.state.tool = 'move';
    expect(describeOverlay(documentOf(['a', 'b']))?.layerOutline).toBeTruthy();
  });

  it('prefers a layer mid-drag over the committed selection', () => {
    harness.state.tool = 'move';
    harness.overrides.set('b', { x: 100, y: 100 });
    const doc = documentOf(['a', 'b']);
    const dragged = describeOverlay(doc)?.layerOutline;
    harness.overrides.clear();
    const committed = describeOverlay(doc)?.layerOutline;
    // 'a' is selected, but 'b' carries the live override, so the marquee tracks
    // 'b' rather than lagging on the selection.
    expect(dragged).not.toEqual(committed);
  });

  it('is absent when the selected layer is gone', () => {
    harness.state.tool = 'move';
    expect(describeOverlay(documentOf([], { selectedLayerId: 'missing' }))?.layerOutline).toBeNull();
  });
});

describe('the transform frame', () => {
  it('is absent for every tool but transform', () => {
    harness.stores.transformSession.set({
      layerId: 'a',
      start: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    } as never);
    expect(describeOverlay().transformFrame).toBeNull();
  });

  it('frames the open session under the transform tool', () => {
    harness.state.tool = 'transform';
    harness.stores.transformSession.set({
      layerId: 'a',
      start: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    } as never);
    expect(describeOverlay().transformFrame).toMatchObject({ corners: expect.any(Array) });
  });

  it('frames the float instead when pixels are lifted', () => {
    harness.state.tool = 'transform';
    harness.float.value = {
      layerId: 'a',
      pixels: { rect: { height: 4, width: 4, x: 0, y: 0 } },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    } as unknown as FloatingSelection;
    // No transform session at all — the float alone drives the frame.
    expect(describeOverlay().transformFrame).toMatchObject({ corners: expect.any(Array) });
  });

  it('is absent when the float layer is gone rather than framed at the origin', () => {
    harness.state.tool = 'transform';
    harness.float.value = {
      layerId: 'missing',
      pixels: { rect: { height: 4, width: 4, x: 0, y: 0 } },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    } as unknown as FloatingSelection;
    expect(describeOverlay().transformFrame).toBeNull();
  });

  it('is absent when the session layer is gone', () => {
    harness.state.tool = 'transform';
    harness.stores.transformSession.set({
      layerId: 'missing',
      start: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    } as never);
    expect(describeOverlay().transformFrame).toBeNull();
  });
});

describe('the marching ants', () => {
  it('are absent without a selection', () => {
    expect(describeOverlay().marchingAnts).toBeNull();
  });

  it('carry the live phase so an overlay-only tick animates them', () => {
    harness.selection.hasSelection.mockReturnValue(true);
    harness.state.phase = 12;
    expect(describeOverlay().marchingAnts).toMatchObject({ phase: 12 });
  });

  it('ride the float matrix so they track lifted pixels, not the layer', () => {
    harness.selection.hasSelection.mockReturnValue(true);
    const ants = [2, 0, 0, 2, 5, 5] as unknown as Mat2d;
    expect(describeOverlay(documentOf(), { ants } as FloatingSelectionFrame).marchingAnts).toMatchObject({
      matrix: ants,
    });
  });

  it('fall back to no matrix when nothing is floating', () => {
    harness.selection.hasSelection.mockReturnValue(true);
    expect(describeOverlay().marchingAnts).toMatchObject({ matrix: null });
  });
});

describe('SAM', () => {
  it('reports no input when the session is not a visual one', () => {
    harness.stores.samInteraction.set({ input: { prompt: 'a cat', type: 'text' } } as never);
    expect(describeOverlay().samInput).toBeNull();
  });

  it('passes a visual session through', () => {
    const input = { points: [], type: 'visual' };
    harness.stores.samInteraction.set({ input } as never);
    expect(describeOverlay().samInput).toBe(input);
  });

  it('draws the preview semi-transparent over the region it describes', () => {
    const sam = {
      data: { id: 'mask' },
      rect: { height: 4, width: 4, x: 1, y: 1 },
    } as unknown as SamPreviewState;
    expect(createOverlayFrame(harness.deps).describe(documentOf(), VIEW, null, sam).samPreview).toEqual({
      opacity: 0.45,
      rect: { height: 4, width: 4, x: 1, y: 1 },
      surface: sam.data,
    });
  });
});

describe('settings pass-through', () => {
  it('forwards each display toggle from its store', () => {
    harness.stores.bboxGrid.set(16);
    harness.stores.bboxOverlay.set(true);
    harness.stores.ruleOfThirds.set(true);
    harness.stores.showGrid.set(true);
    expect(describeOverlay()).toMatchObject({
      bboxOverlay: true,
      gridSize: 16,
      ruleOfThirds: true,
      showGrid: true,
    });
  });

  it('forwards the in-progress tool previews', () => {
    const gradient = { end: { x: 1, y: 1 }, start: { x: 0, y: 0 } };
    const lasso = [{ x: 0, y: 0 }];
    const marquee = { kind: 'rect' as const, rect: { height: 2, width: 2, x: 0, y: 0 } };
    const shape = { kind: 'ellipse' as const, rect: { height: 3, width: 3, x: 1, y: 1 } };
    harness.stores.gradientPreview.set(gradient);
    harness.stores.lassoPreview.set(lasso);
    harness.stores.marqueePreview.set(marquee);
    harness.stores.shapePreview.set(shape);
    expect(describeOverlay()).toMatchObject({
      gradientPreview: gradient,
      lassoPreview: lasso,
      marqueePreview: marquee,
      shapePreview: shape,
    });
  });

  it('forwards the cursor ring and the view transform', () => {
    harness.state.cursor = { point: { x: 3, y: 4 }, radiusDoc: 5 };
    const overlay = describeOverlay();
    expect(overlay.cursor).toEqual({ point: { x: 3, y: 4 }, radiusDoc: 5 });
    expect(overlay.view).toBe(VIEW);
  });
});

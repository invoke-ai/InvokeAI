import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { SamPreviewState } from '@workbench/canvas-engine/controllers/previewStateController';
import type { FloatingSelection } from '@workbench/canvas-engine/selection/floatingSelection';
import type { Mat2d, ToolId } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { FloatingSelectionFrame } from './floatingSelectionFrame';

import { createOverlayFrame, type CreateOverlayFrameDeps } from './overlayFrame';

const VIEW = [1, 0, 0, 1, 0, 0] as unknown as Mat2d;
const SCREEN = { dpr: 1, view: VIEW, viewportSize: { height: 64, width: 64 } };

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
    stacks: stacksFrom(layerIds.map((id) => layer(id) as unknown as CanvasLayerContract)),
    selectedLayerId: layerIds[0] ?? null,
    width: 64,
    ...overrides,
  }) as unknown as CanvasDocumentContractV3;

interface Harness {
  deps: CreateOverlayFrameDeps;
  stores: ReturnType<typeof createEngineStores>;
  overrides: Map<string, { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }>;
  state: {
    tool: ToolId;
    cursor: { point: { x: number; y: number }; radiusDoc: number } | null;
    phase: number;
    pulseTime: number | null;
  };
  float: { value: FloatingSelection | null };
  selection: { hasSelection: ReturnType<typeof vi.fn>; antsPaths: ReturnType<typeof vi.fn> };
}

let harness: Harness;

const makeHarness = (): Harness => {
  const stores = createEngineStores();
  const overrides = new Map<string, { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }>();
  const state: Harness['state'] = { cursor: null, phase: 0, pulseTime: null, tool: 'view' };
  const float: Harness['float'] = { value: null };
  const selection = { antsPaths: vi.fn(() => []), hasSelection: vi.fn(() => false) };
  return {
    deps: {
      getActiveToolId: () => state.tool,
      getAntsPhase: () => state.phase,
      getColorLoupe: () => null,
      getSamPulseTime: () => state.pulseTime,
      getFloatingSelection: () => float.value,
      getOverlayCursor: () => state.cursor,
      getStagedRegions: () => [],
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

const describeOverlay = (
  doc = documentOf(),
  floatFrame: FloatingSelectionFrame | null = null,
  sam: Parameters<ReturnType<typeof createOverlayFrame>['describe']>[3] = null
) => createOverlayFrame(harness.deps).describe(doc, SCREEN, floatFrame, sam);

beforeEach(() => {
  harness = makeHarness();
});

describe('the bbox', () => {
  it('shows the committed frame when nothing is being dragged', () => {
    expect(describeOverlay().bbox).toEqual({ height: 32, width: 32, x: 0, y: 0 });
  });

  it('draws handles only while the bbox tool is active', () => {
    expect(describeOverlay().bboxHandles).toBe(false);
    harness.state.tool = 'bbox';
    expect(describeOverlay().bboxHandles).toBe(true);
  });
});

describe('the move outline', () => {
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
    expect(dragged).not.toEqual(committed);
  });
});

describe('the transform frame', () => {
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
    expect(describeOverlay().transformFrame).toMatchObject({ corners: expect.any(Array) });
  });
});

describe('the marching ants', () => {
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
});

describe('SAM', () => {
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
    expect(createOverlayFrame(harness.deps).describe(documentOf(), SCREEN, null, sam).samPreview).toEqual({
      opacity: 0.45,
      outline: null,
      phase: 0,
      rect: { height: 4, width: 4, x: 1, y: 1 },
      surface: sam.data,
    });
  });
});

describe('the SAM preview', () => {
  const preview = () =>
    ({
      data: { canvas: {}, ctx: {}, height: 4, width: 4 },
      guard: {},
      isolated: false,
      outline: 'outline-path' as unknown as Path2D,
      rect: { height: 4, width: 4, x: 1, y: 2 },
    }) as unknown as NonNullable<Parameters<ReturnType<typeof createOverlayFrame>['describe']>[3]>;

  it('pulses the opacity from the clock on a 2s cycle', () => {
    harness.state.pulseTime = 500;
    expect(describeOverlay(documentOf(), null, preview()).samPreview?.opacity).toBeCloseTo(0.6, 5);
    harness.state.pulseTime = 1500;
    expect(describeOverlay(documentOf(), null, preview()).samPreview?.opacity).toBeCloseTo(0.3, 5);
    harness.state.pulseTime = 1000;
    expect(describeOverlay(documentOf(), null, preview()).samPreview?.opacity).toBeCloseTo(0.45, 5);
  });
});

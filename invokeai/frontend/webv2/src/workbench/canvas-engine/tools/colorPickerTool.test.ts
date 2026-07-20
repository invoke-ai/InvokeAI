import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createLayerCacheStore, type LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { describe, expect, it, vi } from 'vitest';

import { createColorPickerTool } from './colorPickerTool';

/** A mutable RGBA pixel a test can change between gesture steps, injected into every `getImageData` call. */
type MutablePixel = { current: readonly [number, number, number, number] };

/** A minimal `RasterBackend` whose scratch surfaces report `pixel.current` for every `getImageData` call. */
const createFixedPixelBackend = (pixel: MutablePixel): RasterBackend => ({
  createImageBitmap: () => Promise.resolve({} as ImageBitmap),
  createSurface: (width: number, height: number): RasterSurface => {
    const canvas = { height, width } as unknown as OffscreenCanvas;
    const ctx = {
      clearRect: () => {},
      drawImage: () => {},
      getImageData: () =>
        ({ data: Uint8ClampedArray.from(pixel.current), height: 1, width: 1 }) as unknown as ImageData,
      restore: () => {},
      save: () => {},
      setTransform: () => {},
    } as unknown as OffscreenCanvasRenderingContext2D;
    return { canvas, ctx, height, resize: () => {}, width };
  },
  encodeSurface: () => Promise.resolve(new Blob()),
});

const paintLayer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const makeDoc = (): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers: [paintLayer('paint1')],
  selectedLayerId: 'paint1',
  version: 2,
  width: 100,
});

const pointer = (x: number, y: number, opts: { buttons?: number } = {}): PointerInput => ({
  buttons: opts.buttons ?? 1,
  documentPoint: { x, y },
  modifiers: { alt: true, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: 0,
});

interface Harness {
  ctx: ToolContext;
  pixel: MutablePixel;
  layers: LayerCacheStore;
  dispatched: CanvasProjectMutation[];
  strokes: unknown[];
  overlayCursors: unknown[];
}

const createHarness = (doc: CanvasDocumentContractV2 | null): Harness => {
  const pixel: MutablePixel = { current: [10, 20, 30, 255] };
  const backend = createFixedPixelBackend(pixel);
  const layers = createLayerCacheStore(backend);
  if (doc) {
    layers.getOrCreate('paint1', doc.width, doc.height);
  }
  const stores = createEngineStores();
  const dispatched: CanvasProjectMutation[] = [];
  const strokes: unknown[] = [];
  const overlayCursors: unknown[] = [];

  const ctx: ToolContext = {
    backend,
    commitStructural: vi.fn(),
    createLayerId: () => 'unused',
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    dispatch: (action) => dispatched.push(action),
    emitStrokeCommitted: (event) => strokes.push(event),
    getDocument: () => doc,
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted: vi.fn(),
    setLayerTransformOverride: vi.fn(),
    setOverlayCursor: (cursor) => overlayCursors.push(cursor),
    stores,
    updateCursor: vi.fn(),
    viewport: { getZoom: () => 1 } as unknown as ToolContext['viewport'],
  };

  return { ctx, dispatched, layers, overlayCursors, pixel, strokes };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);

describe('color picker tool', () => {
  it('samples the composited color on pointer down and writes it into brushOptions.color', () => {
    const h = createHarness(makeDoc());
    const tool = createColorPickerTool();

    down(tool, h.ctx, pointer(10, 10));

    expect(h.ctx.stores.brushOptions.get().color).toBe('#0a141e');
  });

  it('re-samples on drag (primary button held) as the sample changes', () => {
    const h = createHarness(makeDoc());
    const tool = createColorPickerTool();

    down(tool, h.ctx, pointer(10, 10));
    expect(h.ctx.stores.brushOptions.get().color).toBe('#0a141e');

    h.pixel.current = [40, 50, 60, 255];
    move(tool, h.ctx, pointer(20, 20));
    expect(h.ctx.stores.brushOptions.get().color).toBe('#28323c');
  });

  it('does not sample on move when no button is held', () => {
    const h = createHarness(makeDoc());
    const tool = createColorPickerTool();
    const defaultColor = h.ctx.stores.brushOptions.get().color;

    move(tool, h.ctx, pointer(10, 10, { buttons: 0 }));

    expect(h.ctx.stores.brushOptions.get().color).toBe(defaultColor);
  });

  it('leaves brushOptions untouched when the point falls outside the document', () => {
    const h = createHarness(makeDoc());
    const tool = createColorPickerTool();
    const defaultColor = h.ctx.stores.brushOptions.get().color;

    down(tool, h.ctx, pointer(-5, 10));

    expect(h.ctx.stores.brushOptions.get().color).toBe(defaultColor);
  });

  it('leaves brushOptions untouched when there is no document', () => {
    const h = createHarness(null);
    const tool = createColorPickerTool();
    const defaultColor = h.ctx.stores.brushOptions.get().color;

    down(tool, h.ctx, pointer(10, 10));

    expect(h.ctx.stores.brushOptions.get().color).toBe(defaultColor);
  });

  it('never dispatches and never emits a committed stroke', () => {
    const h = createHarness(makeDoc());
    const tool = createColorPickerTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(20, 20));
    up(tool, h.ctx, pointer(20, 20));

    expect(h.dispatched).toHaveLength(0);
    expect(h.strokes).toHaveLength(0);
  });

  it('shows a ring cursor while active and clears it on deactivate', () => {
    const h = createHarness(makeDoc());
    const tool = createColorPickerTool();

    move(tool, h.ctx, pointer(10, 10, { buttons: 0 }));
    expect(h.overlayCursors.at(-1)).toMatchObject({ point: { x: 10, y: 10 } });

    tool.onDeactivate?.(h.ctx);
    expect(h.overlayCursors.at(-1)).toBeNull();
  });

  it('reports a crosshair cursor', () => {
    const tool = createColorPickerTool();
    expect(tool.cursor?.({} as ToolContext)).toBe('crosshair');
  });
});

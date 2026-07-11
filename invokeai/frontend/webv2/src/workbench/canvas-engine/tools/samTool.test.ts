import type { SamSessionSnapshot } from '@workbench/canvas-engine/engineStores';
import type { PointerInput } from '@workbench/canvas-engine/types';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { describe, expect, it, vi } from 'vitest';

import type { ToolContext } from './tool';

import { createSamTool } from './samTool';

const sourceRect = { height: 100, width: 100, x: 0, y: 0 };

const visualSnapshot = (overrides: Partial<SamSessionSnapshot> = {}): SamSessionSnapshot => ({
  error: null,
  hasPreview: false,
  input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
  isolatedPreview: true,
  layerId: 'source',
  pointLabel: 'include',
  sourceRect,
  status: 'ready',
  ...overrides,
});

const pointer = (x: number, y: number, options: { buttons?: number; shift?: boolean } = {}): PointerInput => ({
  buttons: options.buttons ?? 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: options.shift ?? false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: 0,
});

const createHarness = (snapshot = visualSnapshot()) => {
  const stores = createEngineStores();
  stores.samSession.set(snapshot);
  const invalidate = vi.fn();
  const updateSamInput = vi.fn((input: SamSessionSnapshot['input']) => {
    const current = stores.samSession.get();
    if (current) {
      stores.samSession.set({ ...current, input });
    }
  });
  const ctx = {
    invalidate,
    stores,
    updateCursor: vi.fn(),
    updateSamInput,
    viewport: {
      documentToScreen: (point: { x: number; y: number }) => point,
      viewMatrix: () => ({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }),
    },
  } as unknown as ToolContext;
  return { ctx, invalidate, stores, tool: createSamTool(), updateSamInput };
};

const down = (h: ReturnType<typeof createHarness>, input: PointerInput): void => h.tool.onPointerDown?.(h.ctx, input);
const move = (h: ReturnType<typeof createHarness>, input: PointerInput): void =>
  h.tool.onPointerMove?.(h.ctx, input, [input]);
const up = (h: ReturnType<typeof createHarness>, input: PointerInput): void => h.tool.onPointerUp?.(h.ctx, input);

describe('createSamTool', () => {
  it('adds configured points on click and temporarily flips their label with Shift', () => {
    const h = createHarness();

    down(h, pointer(20, 30));
    up(h, pointer(20, 30, { buttons: 0 }));
    down(h, pointer(50, 60, { shift: true }));
    up(h, pointer(50, 60, { buttons: 0, shift: true }));

    expect(h.stores.samSession.get()?.input).toEqual({
      bbox: null,
      excludePoints: [{ x: 50, y: 60 }],
      includePoints: [{ x: 20, y: 30 }],
      type: 'visual',
    });
    expect(h.invalidate.mock.calls.every(([payload]) => payload.overlay === true && !payload.all)).toBe(true);
  });

  it('removes an existing point on click but drags it after the screen threshold', () => {
    const h = createHarness(
      visualSnapshot({
        input: { bbox: null, excludePoints: [], includePoints: [{ x: 20, y: 20 }], type: 'visual' },
      })
    );

    down(h, pointer(20, 20));
    up(h, pointer(20, 20, { buttons: 0 }));
    expect(h.stores.samSession.get()?.input.includePoints).toEqual([]);

    h.stores.samSession.set(
      visualSnapshot({ input: { bbox: null, excludePoints: [], includePoints: [{ x: 20, y: 20 }], type: 'visual' } })
    );
    down(h, pointer(20, 20));
    move(h, pointer(21, 21));
    expect(h.stores.samSession.get()?.input.includePoints).toEqual([{ x: 20, y: 20 }]);
    move(h, pointer(35, 40));
    up(h, pointer(35, 40, { buttons: 0 }));
    expect(h.stores.samSession.get()?.input.includePoints).toEqual([{ x: 35, y: 40 }]);
  });

  it('creates a clipped bbox from an empty drag and treats a sub-threshold drag as a click', () => {
    const h = createHarness();

    down(h, pointer(80, 80));
    move(h, pointer(120, 130));
    up(h, pointer(120, 130, { buttons: 0 }));
    expect(h.stores.samSession.get()?.input.bbox).toEqual({ height: 20, width: 20, x: 80, y: 80 });

    down(h, pointer(10, 10));
    move(h, pointer(12, 12));
    up(h, pointer(12, 12, { buttons: 0 }));
    expect(h.stores.samSession.get()?.input.includePoints).toEqual([{ x: 12, y: 12 }]);
  });

  it('does not publish a degenerate bbox when a boundary drag clips to zero area', () => {
    const h = createHarness();

    down(h, pointer(100, 100));
    move(h, pointer(120, 130));
    up(h, pointer(120, 130, { buttons: 0 }));

    expect(h.stores.samSession.get()?.input.bbox).toBeNull();
  });

  it('accepts and canonicalizes a near-edge point but rejects the exact right and bottom edges', () => {
    const h = createHarness();

    down(h, pointer(99.8, 99.6));
    up(h, pointer(99.8, 99.6, { buttons: 0 }));
    down(h, pointer(100, 50));
    up(h, pointer(100, 50, { buttons: 0 }));
    down(h, pointer(50, 100));
    up(h, pointer(50, 100, { buttons: 0 }));

    expect(h.stores.samSession.get()?.input.includePoints).toEqual([{ x: 99, y: 99 }]);
  });

  it('moves the bbox body and resizes a handle while clipping to source bounds', () => {
    const h = createHarness(
      visualSnapshot({
        input: { bbox: { height: 30, width: 40, x: 20, y: 20 }, excludePoints: [], includePoints: [], type: 'visual' },
      })
    );

    down(h, pointer(40, 35));
    move(h, pointer(90, 90));
    up(h, pointer(90, 90, { buttons: 0 }));
    expect(h.stores.samSession.get()?.input.bbox).toEqual({ height: 30, width: 40, x: 60, y: 70 });

    down(h, pointer(100, 100));
    move(h, pointer(150, 150));
    up(h, pointer(150, 150, { buttons: 0 }));
    expect(h.stores.samSession.get()?.input.bbox).toEqual({ height: 30, width: 40, x: 60, y: 70 });
  });

  it('restores the exact pre-gesture input on pointercancel', () => {
    const initial = visualSnapshot({
      input: { bbox: null, excludePoints: [{ x: 10, y: 10 }], includePoints: [], type: 'visual' },
    });
    const h = createHarness(initial);

    down(h, pointer(10, 10));
    move(h, pointer(50, 50));
    expect(h.stores.samSession.get()?.input.excludePoints).toEqual([{ x: 50, y: 50 }]);
    h.tool.onPointerCancel?.(h.ctx);

    expect(h.stores.samSession.get()?.input).toEqual(initial.input);
  });
});

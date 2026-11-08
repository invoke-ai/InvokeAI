import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { ShapeToolKind } from '@workbench/canvas-engine/engineStores';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createRecordingStrokeEdit } from '@workbench/canvas-engine/tools/strokeEdit.testStub';
import { describe, expect, it, vi } from 'vitest';

import { createShapeTool } from './shapeTool';

const pointer = (x: number, y: number, timeStamp = 0, ctrl = false): PointerInput => ({
  buttons: 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp,
});

/**
 * Draws `kind` over document 20..80 × 20..60 (the polygon and freehand trace its right triangle). A cutout holds Ctrl
 * on the commit: the release, or the polygon's closing press on its first vertex.
 */
const draw = (tool: Tool, ctx: ToolContext, kind: ShapeToolKind, cutout = false): void => {
  if (kind === 'polygon') {
    for (const [x, y, t] of [
      [20, 20, 0],
      [80, 20, 1000],
      [80, 60, 2000],
    ] as const) {
      tool.onPointerDown?.(ctx, pointer(x, y, t));
      tool.onPointerUp?.(ctx, pointer(x, y, t));
    }
    tool.onPointerDown?.(ctx, pointer(20, 20, 3000, cutout));
    tool.onPointerUp?.(ctx, pointer(20, 20, 3000, cutout));
    return;
  }
  tool.onPointerDown?.(ctx, pointer(20, 20));
  for (const point of [pointer(80, 20), pointer(80, 60)]) {
    tool.onPointerMove?.(ctx, point, [point]);
  }
  tool.onPointerUp?.(ctx, pointer(80, 60, 0, cutout));
};

/** Draws (or with `cutout`, cuts) `kind` on the selected `layer`, whose cache starts as `prefill` when given. */
const drawOn = (
  layer: CanvasLayerContract,
  kind: ShapeToolKind,
  { cutout = false, prefill, strokeOnly = false }: { cutout?: boolean; prefill?: string; strokeOnly?: boolean } = {}
) => {
  const backend = createDomRasterBackend();
  const layers = createLayerCacheStore(backend);
  if (prefill) {
    const entry = layers.getOrCreateRect(layer.id, { height: 100, width: 100, x: 0, y: 0 });
    entry.stale = false;
    entry.surface.ctx.fillStyle = prefill;
    entry.surface.ctx.fillRect(0, 0, 100, 100);
  }
  const stores = createEngineStores();
  stores.shapeOptions.set({ ...stores.shapeOptions.get(), fillEnabled: !strokeOnly, kind, strokeEnabled: strokeOnly });
  // A translucent foreground must still paint full coverage.
  stores.colorPair.set({ background: '#000000', foreground: '#ff000040' });
  const strokeEdit = createRecordingStrokeEdit();
  const addedLayers: unknown[] = [];
  const ctx: ToolContext = {
    backend,
    beginStrokeEdit: () => strokeEdit.edit,
    captureInsertionAnchor: createTestInsertionAnchorCapture('p'),
    commitStructural: (_label, forward) => {
      addedLayers.push(forward);
      return { status: 'committed' as const };
    },
    createLayerId: () => 'created',
    createPath2D: (d) => new Path2D(d),
    dispatch: vi.fn(),
    getDocument: () => documentFrom([layer], layer.id),
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted: vi.fn(),
    scheduleFrame: () => () => undefined,
    setLayerTransformOverride: vi.fn(),
    setOverlayCursor: vi.fn(),
    stores,
    updateCursor: vi.fn(),
    viewport: { documentToScreen: (p: { x: number; y: number }) => p } as unknown as Viewport,
  };
  // Off macOS, Ctrl is the cutout key.
  draw(createShapeTool({ isMacOs: false }), ctx, kind, cutout);
  const entry = layers.get(layer.id);
  /** The layer's alpha at a layer-local pixel, or 0 outside its cache. */
  const alphaAt = (x: number, y: number): number => {
    if (!entry || x < entry.rect.x || y < entry.rect.y) {
      return 0;
    }
    const pixel = entry.surface.ctx.getImageData(x - entry.rect.x, y - entry.rect.y, 1, 1).data;
    return pixel[3] ?? 0;
  };
  return { addedLayers, alphaAt, strokeEdit };
};

describe('shape tool on a mask in Chromium', () => {
  it.each(['rect', 'ellipse', 'polygon', 'freehand'] as const)(
    'covers a %s with opaque mask alpha inside the shape and nothing outside it',
    (kind) => {
      const { addedLayers, alphaAt, strokeEdit } = drawOn(layerContract('mask', 'inpaint_mask'), kind);

      expect(addedLayers).toHaveLength(0);
      expect(strokeEdit.record.commits).toHaveLength(1);
      // Inside every kind: near the right edge, vertically centred (inside the triangle too).
      expect(alphaAt(72, 40)).toBe(255);
      // Outside every kind: past the right edge, and the triangle's empty lower-left corner.
      expect(alphaAt(90, 40)).toBe(0);
      if (kind === 'polygon' || kind === 'freehand') {
        expect(alphaAt(25, 55)).toBe(0);
      } else {
        expect(alphaAt(50, 40)).toBe(255);
      }
    }
  );

  it('paints an inside stroke without filling the mask interior', () => {
    const { alphaAt, strokeEdit } = drawOn(layerContract('mask', 'inpaint_mask'), 'rect', { strokeOnly: true });
    expect(alphaAt(22, 40)).toBe(255);
    expect(alphaAt(50, 40)).toBe(0);
    expect(alphaAt(19, 40)).toBe(0);
    expect(strokeEdit.record.commits).toHaveLength(1);
  });

  it('lands coverage under the cursor on a moved and scaled regional mask', () => {
    const mask = layerContract('region', 'regional_guidance', {
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 10, y: 10 },
    });
    const { alphaAt } = drawOn(mask, 'rect');

    // Document (72, 40) is local ((72 - 10) / 2, (40 - 10) / 2) = (31, 15).
    expect(alphaAt(31, 15)).toBe(255);
    // Document (90, 40) is local (40, 15): outside the shape.
    expect(alphaAt(40, 15)).toBe(0);
  });
});

describe('shape cutout in Chromium', () => {
  it.each([
    ['paint layer', 'raster', 'rect'],
    ['paint layer', 'raster', 'polygon'],
    ['mask', 'inpaint_mask', 'ellipse'],
    ['mask', 'regional_guidance', 'freehand'],
  ] as const)('clears a %s (%s) inside a %s and keeps every pixel outside it', (_label, type, kind) => {
    const { addedLayers, alphaAt, strokeEdit } = drawOn(layerContract('target', type), kind, {
      cutout: true,
      prefill: '#336699',
    });

    expect(addedLayers).toHaveLength(0);
    expect(strokeEdit.record.commits).toHaveLength(1);
    const event = strokeEdit.record.commits[0]!;
    expect(event.tool).toBe('shape-cutout');
    // The before image is the undo: the pixels the cutout removed, still opaque there.
    expect(event.beforeImageData.data[3]).toBe(255);
    expect(alphaAt(72, 40)).toBe(0);
    expect(alphaAt(90, 40)).toBe(255);
    expect(alphaAt(10, 10)).toBe(255);
    if (kind === 'polygon' || kind === 'freehand') {
      expect(alphaAt(25, 55)).toBe(255);
    } else {
      expect(alphaAt(50, 40)).toBe(0);
    }
  });

  it('cuts only the inside polygon outline and leaves its hollow interior intact', () => {
    const { alphaAt, strokeEdit } = drawOn(layerContract('paint', 'raster'), 'polygon', {
      cutout: true,
      prefill: '#336699',
      strokeOnly: true,
    });
    expect(alphaAt(76, 35)).toBe(0);
    expect(alphaAt(65, 30)).toBe(255);
    expect(alphaAt(25, 55)).toBe(255);
    expect(strokeEdit.record.commits).toHaveLength(1);
    expect(strokeEdit.record.commits[0]?.tool).toBe('shape-cutout');
  });
});

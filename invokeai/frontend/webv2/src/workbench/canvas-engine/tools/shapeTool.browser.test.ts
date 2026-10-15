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

const pointer = (x: number, y: number, timeStamp = 0): PointerInput => ({
  buttons: 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp,
});

/** Draws `kind` over document 20..80 × 20..60 (the polygon and freehand trace its right triangle). */
const draw = (tool: Tool, ctx: ToolContext, kind: ShapeToolKind): void => {
  if (kind === 'polygon') {
    for (const [x, y, t] of [
      [20, 20, 0],
      [80, 20, 1000],
      [80, 60, 2000],
    ] as const) {
      tool.onPointerDown?.(ctx, pointer(x, y, t));
      tool.onPointerUp?.(ctx, pointer(x, y, t));
    }
    tool.onKeyCommand?.(ctx, 'apply');
    return;
  }
  tool.onPointerDown?.(ctx, pointer(20, 20));
  for (const point of [pointer(80, 20), pointer(80, 60)]) {
    tool.onPointerMove?.(ctx, point, [point]);
  }
  tool.onPointerUp?.(ctx, pointer(80, 60));
};

const paintMask = (mask: CanvasLayerContract, kind: ShapeToolKind) => {
  const backend = createDomRasterBackend();
  const layers = createLayerCacheStore(backend);
  const stores = createEngineStores();
  stores.shapeOptions.set({ ...stores.shapeOptions.get(), kind });
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
    getDocument: () => documentFrom([mask], mask.id),
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
  draw(createShapeTool(), ctx, kind);
  const entry = layers.get(mask.id);
  /** The mask's alpha at a layer-local pixel, or 0 outside its cache. */
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
      const { addedLayers, alphaAt, strokeEdit } = paintMask(layerContract('mask', 'inpaint_mask'), kind);

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

  it('lands coverage under the cursor on a moved and scaled regional mask', () => {
    const mask = layerContract('region', 'regional_guidance', {
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 10, y: 10 },
    });
    const { alphaAt } = paintMask(mask, 'rect');

    // Document (72, 40) is local ((72 - 10) / 2, (40 - 10) / 2) = (31, 15).
    expect(alphaAt(31, 15)).toBe(255);
    // Document (90, 40) is local (40, 15): outside the shape.
    expect(alphaAt(40, 15)).toBe(0);
  });
});

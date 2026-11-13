import type { ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createSelectionState } from '@workbench/canvas-engine/selection/selectionState';
import { createLassoTool } from '@workbench/canvas-engine/tools/lassoTool';
import { expect, it } from 'vitest';

it.each([
  [1, 1, 1],
  [10, 10, 0.1],
  [0.5, 0.5, 10],
  [8, 0.25, 1],
])('rasterizes a %s by %s lasso at %sx zoom', (width, height, zoom) => {
  const selection = createSelectionState({
    backend: createDomRasterBackend(),
    createPath2D: (d) => new Path2D(d),
    getDocumentSize: () => ({ height: 100, width: 100 }),
    onChange: () => undefined,
  });
  const ctx = {
    commitSelection: selection.commit,
    createPath2D: (d?: string) => new Path2D(d),
    invalidate: () => {},
    stores: createEngineStores(),
  } as unknown as ToolContext;
  const pointer = (x: number, y: number): PointerInput => ({
    buttons: 1,
    documentPoint: { x, y },
    modifiers: { alt: false, ctrl: false, meta: false, shift: false },
    pointerType: 'mouse',
    pressure: 0.5,
    screenPoint: { x: x * zoom, y: y * zoom },
    timeStamp: 0,
  });
  const tool = createLassoTool();
  try {
    tool.onPointerDown?.(ctx, pointer(10, 10));
    for (const [x, y] of [
      [10 + width, 10],
      [10 + width, 10 + height],
      [10, 10 + height],
    ] as const) {
      const input = pointer(x, y);
      tool.onPointerMove?.(ctx, input, [input]);
    }
    tool.onPointerUp?.(ctx, pointer(10, 10));

    expect(selection.hasSelection()).toBe(true);
    const mask = selection.mask()!;
    const { data } = mask.surface.ctx.getImageData(0, 0, mask.surface.width, mask.surface.height);
    expect([...data].some((value, index) => index % 4 === 3 && value > 0)).toBe(true);
    expect(selection.containsPoint({ x: 10 + width / 2, y: 10 + height / 2 })).toBe(true);
  } finally {
    selection.dispose();
  }
});

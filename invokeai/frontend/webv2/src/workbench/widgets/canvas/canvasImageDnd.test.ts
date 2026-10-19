import { describe, expect, it } from 'vitest';

import { getCanvasImageDropData, isCanvasImageDropData } from './canvasImageDnd';

describe('canvasImageDnd', () => {
  it.each(['raster', 'control', 'regional-reference', 'inpaint-mask', 'control-resized'] as const)(
    'round-trips the %s destination through the factory and guard',
    (destination) => {
      const data = getCanvasImageDropData(destination);

      expect(data).toEqual({ destination, kind: 'canvas-image-target' });
      expect(isCanvasImageDropData(data)).toBe(true);
    }
  );

  it.each([
    null,
    undefined,
    'canvas-image-target',
    {},
    { destination: 'raster' },
    { kind: 'canvas-image-target' },
    { destination: 'regional-guidance', kind: 'canvas-image-target' },
    { destination: 'unknown', kind: 'canvas-image-target' },
    { destination: null, kind: 'canvas-image-target' },
  ])('rejects malformed canvas image drop data: %j', (value) => {
    expect(isCanvasImageDropData(value)).toBe(false);
  });
});

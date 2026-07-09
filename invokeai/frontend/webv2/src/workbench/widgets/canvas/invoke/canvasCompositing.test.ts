import { describe, expect, it } from 'vitest';

import { DEFAULT_CANVAS_COMPOSITING, readCanvasCompositingSettings } from './canvasCompositing';

describe('readCanvasCompositingSettings', () => {
  it('returns legacy defaults for undefined values', () => {
    expect(readCanvasCompositingSettings(undefined)).toEqual(DEFAULT_CANVAS_COMPOSITING);
  });

  it('mirrors the legacy params-slice defaults exactly', () => {
    expect(DEFAULT_CANVAS_COMPOSITING).toEqual({
      coherenceEdgeSize: 16,
      coherenceMinDenoise: 0,
      coherenceMode: 'Gaussian Blur',
      infillColorValue: { a: 1, b: 0, g: 0, r: 0 },
      infillMethod: 'lama',
      infillPatchmatchDownscaleSize: 1,
      infillTileSize: 32,
      maskBlur: 16,
    });
  });

  it('reads valid persisted values', () => {
    const settings = readCanvasCompositingSettings({
      coherenceEdgeSize: 32,
      coherenceMinDenoise: 0.2,
      coherenceMode: 'Box Blur',
      infillMethod: 'patchmatch',
      maskBlur: 8,
    });
    expect(settings.coherenceEdgeSize).toBe(32);
    expect(settings.coherenceMinDenoise).toBe(0.2);
    expect(settings.coherenceMode).toBe('Box Blur');
    expect(settings.infillMethod).toBe('patchmatch');
    expect(settings.maskBlur).toBe(8);
  });

  it('falls back to defaults for invalid enum values', () => {
    const settings = readCanvasCompositingSettings({ coherenceMode: 'Wobble', infillMethod: 'magic' });
    expect(settings.coherenceMode).toBe('Gaussian Blur');
    expect(settings.infillMethod).toBe('lama');
  });

  it('clamps numeric ranges and rounds integers', () => {
    const settings = readCanvasCompositingSettings({
      coherenceEdgeSize: -5,
      coherenceMinDenoise: 5,
      maskBlur: 12.7,
    });
    expect(settings.coherenceEdgeSize).toBe(0);
    expect(settings.coherenceMinDenoise).toBe(1);
    expect(settings.maskBlur).toBe(13);
  });
});

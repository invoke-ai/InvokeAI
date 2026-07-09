import { describe, expect, it } from 'vitest';

import { isCanvasStagingActive, isCanvasToolEnabled } from './canvasInteractionLock';

describe('canvas interaction lock', () => {
  it('treats pending staged candidates and in-flight canvas generation as active staging', () => {
    expect(isCanvasStagingActive({ hasStagedCandidates: true, isCanvasGenerationInFlight: false })).toBe(true);
    expect(isCanvasStagingActive({ hasStagedCandidates: false, isCanvasGenerationInFlight: true })).toBe(true);
    expect(isCanvasStagingActive({ hasStagedCandidates: false, isCanvasGenerationInFlight: false })).toBe(false);
  });

  it('allows only the view tool while staging locks canvas interactions', () => {
    expect(isCanvasToolEnabled('view', true)).toBe(true);
    expect(isCanvasToolEnabled('bbox', true)).toBe(false);
    expect(isCanvasToolEnabled('brush', true)).toBe(false);
    expect(isCanvasToolEnabled('move', true)).toBe(false);
    expect(isCanvasToolEnabled('colorPicker', true)).toBe(false);
  });

  it('allows all tools when staging is inactive', () => {
    expect(isCanvasToolEnabled('bbox', false)).toBe(true);
    expect(isCanvasToolEnabled('brush', false)).toBe(true);
    expect(isCanvasToolEnabled('colorPicker', false)).toBe(true);
  });
});

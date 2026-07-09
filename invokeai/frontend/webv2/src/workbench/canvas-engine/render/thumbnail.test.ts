import { describe, expect, it } from 'vitest';

import { fitThumbnailSize } from './thumbnail';

describe('fitThumbnailSize', () => {
  it('scales a large landscape surface to fit the box width', () => {
    expect(fitThumbnailSize(200, 100, 96)).toEqual({ height: 48, width: 96 });
  });

  it('scales a large portrait surface to fit the box height', () => {
    expect(fitThumbnailSize(100, 200, 96)).toEqual({ height: 96, width: 48 });
  });

  it('keeps a square surface square', () => {
    expect(fitThumbnailSize(300, 300, 96)).toEqual({ height: 96, width: 96 });
  });

  it('never upscales a surface smaller than the box', () => {
    expect(fitThumbnailSize(40, 20, 96)).toEqual({ height: 20, width: 40 });
  });

  it('clamps a fitted dimension to at least 1px', () => {
    expect(fitThumbnailSize(1000, 1, 96)).toEqual({ height: 1, width: 96 });
  });

  it('returns a zero size for a degenerate source or box', () => {
    expect(fitThumbnailSize(0, 100, 96)).toEqual({ height: 0, width: 0 });
    expect(fitThumbnailSize(100, 100, 0)).toEqual({ height: 0, width: 0 });
  });
});

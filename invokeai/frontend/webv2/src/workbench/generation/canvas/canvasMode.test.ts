import { describe, expect, it } from 'vitest';

import type { CanvasModeInput } from './canvasMode';
import type { Rect } from './types';

import { detectCanvasMode } from './canvasMode';

const BBOX: Rect = { height: 100, width: 100, x: 0, y: 0 };

const input = (overrides: Partial<CanvasModeInput> = {}): CanvasModeInput => ({
  bbox: BBOX,
  bboxFullyCovered: false,
  contentBounds: null,
  hasActiveInpaintMask: false,
  ...overrides,
});

describe('detectCanvasMode', () => {
  describe('txt2img', () => {
    it('returns txt2img when there is no content at all', () => {
      expect(detectCanvasMode(input({ contentBounds: null }))).toBe('txt2img');
    });

    it('returns txt2img when content lies entirely outside the bbox', () => {
      const contentBounds: Rect = { height: 50, width: 50, x: 200, y: 200 };
      expect(detectCanvasMode(input({ contentBounds }))).toBe('txt2img');
    });

    it('treats a flush edge (0px interior overlap) as no intersection → txt2img', () => {
      // Content's left edge sits exactly on the bbox's right edge (x = 100).
      const contentBounds: Rect = { height: 100, width: 50, x: 100, y: 0 };
      expect(detectCanvasMode(input({ contentBounds }))).toBe('txt2img');
    });

    it('returns txt2img for a zero-width bbox even when content overlaps its line', () => {
      const bbox: Rect = { height: 100, width: 0, x: 0, y: 0 };
      const contentBounds: Rect = { height: 100, width: 100, x: 0, y: 0 };
      expect(detectCanvasMode(input({ bbox, bboxFullyCovered: true, contentBounds }))).toBe('txt2img');
    });

    it('returns txt2img for a zero-height bbox', () => {
      const bbox: Rect = { height: 0, width: 100, x: 0, y: 0 };
      const contentBounds: Rect = { height: 100, width: 100, x: 0, y: 0 };
      expect(detectCanvasMode(input({ bbox, bboxFullyCovered: true, contentBounds }))).toBe('txt2img');
    });

    it('stays txt2img for a new canvas seeded with only an empty inpaint mask', () => {
      // A brand-new canvas carries one empty inpaint mask (see
      // `createNewCanvasStateV2`). An empty mask contributes no raster content
      // (`contentBounds` stays null) and has no content (`hasActiveInpaintMask`
      // is false), so mode detection must NOT flip to inpaint.
      expect(detectCanvasMode(input({ contentBounds: null, hasActiveInpaintMask: false }))).toBe('txt2img');
    });
  });

  describe('outpaint', () => {
    it('returns outpaint when content intersects but does not fully cover the bbox', () => {
      const contentBounds: Rect = { height: 100, width: 50, x: 0, y: 0 };
      expect(detectCanvasMode(input({ bboxFullyCovered: false, contentBounds }))).toBe('outpaint');
    });

    it('counts a 1px interior overlap as intersecting → outpaint', () => {
      // Content's left edge at x = 99: a 1px column overlaps the bbox interior.
      const contentBounds: Rect = { height: 100, width: 50, x: 99, y: 0 };
      expect(detectCanvasMode(input({ bboxFullyCovered: false, contentBounds }))).toBe('outpaint');
    });

    it('returns outpaint even with an active mask when coverage is incomplete', () => {
      const contentBounds: Rect = { height: 100, width: 50, x: 0, y: 0 };
      expect(detectCanvasMode(input({ bboxFullyCovered: false, contentBounds, hasActiveInpaintMask: true }))).toBe(
        'outpaint'
      );
    });
  });

  describe('img2img', () => {
    it('returns img2img when content fully covers the bbox and there is no mask', () => {
      const contentBounds: Rect = { height: 100, width: 100, x: 0, y: 0 };
      expect(detectCanvasMode(input({ bboxFullyCovered: true, contentBounds, hasActiveInpaintMask: false }))).toBe(
        'img2img'
      );
    });

    it('returns img2img when content extends past the bbox but covers it fully', () => {
      const contentBounds: Rect = { height: 200, width: 200, x: -50, y: -50 };
      expect(detectCanvasMode(input({ bboxFullyCovered: true, contentBounds }))).toBe('img2img');
    });
  });

  describe('inpaint', () => {
    it('returns inpaint when content fully covers the bbox and a mask is active', () => {
      const contentBounds: Rect = { height: 100, width: 100, x: 0, y: 0 };
      expect(detectCanvasMode(input({ bboxFullyCovered: true, contentBounds, hasActiveInpaintMask: true }))).toBe(
        'inpaint'
      );
    });
  });
});

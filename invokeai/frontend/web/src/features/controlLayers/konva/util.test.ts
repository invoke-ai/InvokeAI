import { roundUpToMultiple } from 'common/util/roundDownToMultiple';
import { fitRectToGrid, getPrefixedId, getRectUnion } from 'features/controlLayers/konva/util';
import type { Rect } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

describe('util', () => {
  describe('getPrefixedId', () => {
    it('should return a prefixed id', () => {
      expect(getPrefixedId('foo').split(':')[0]).toBe('foo');
    });
  });

  describe('getRectUnion', () => {
    it('should return the union of rects (2 rects)', () => {
      const rect1 = { x: 0, y: 0, width: 10, height: 10 };
      const rect2 = { x: 5, y: 5, width: 10, height: 10 };
      const union = getRectUnion(rect1, rect2);
      expect(union).toEqual({ x: 0, y: 0, width: 15, height: 15 });
    });
    it('should return the union of rects (3 rects)', () => {
      const rect1 = { x: 0, y: 0, width: 10, height: 10 };
      const rect2 = { x: 5, y: 5, width: 10, height: 10 };
      const rect3 = { x: 10, y: 10, width: 10, height: 10 };
      const union = getRectUnion(rect1, rect2, rect3);
      expect(union).toEqual({ x: 0, y: 0, width: 20, height: 20 });
    });
    it('should return the union of rects (2 rects none from zero)', () => {
      const rect1 = { x: 5, y: 5, width: 10, height: 10 };
      const rect2 = { x: 10, y: 10, width: 10, height: 10 };
      const union = getRectUnion(rect1, rect2);
      expect(union).toEqual({ x: 5, y: 5, width: 15, height: 15 });
    });
    it('should return the union of rects (2 rects with negative x/y)', () => {
      const rect1 = { x: -5, y: -5, width: 10, height: 10 };
      const rect2 = { x: 0, y: 0, width: 10, height: 10 };
      const union = getRectUnion(rect1, rect2);
      expect(union).toEqual({ x: -5, y: -5, width: 15, height: 15 });
    });
    it('should return the union of the first rect if only one rect is provided', () => {
      const rect = { x: 0, y: 0, width: 10, height: 10 };
      const union = getRectUnion(rect);
      expect(union).toEqual(rect);
    });
    it('should fall back on an empty rect if no rects are provided', () => {
      const union = getRectUnion();
      expect(union).toEqual({ x: 0, y: 0, width: 0, height: 0 });
    });
  });

  describe('fitRectToGrid', () => {
    it('should fit rect within grid without exceeding bounds', () => {
      const rect: Rect = { x: 0, y: 0, width: 1047, height: 1758 };
      const gridSize = 50;
      const result = fitRectToGrid(rect, gridSize);

      expect(result.x).toBe(roundUpToMultiple(rect.x, gridSize));
      expect(result.y).toBe(roundUpToMultiple(rect.y, gridSize));
      expect(result.width).toBeLessThanOrEqual(rect.width);
      expect(result.height).toBeLessThanOrEqual(rect.height);
      expect(result.width % gridSize).toBe(0);
      expect(result.height % gridSize).toBe(0);
    });

    it('should handle small rect within grid bounds', () => {
      const rect: Rect = { x: 20, y: 30, width: 80, height: 90 };
      const gridSize = 25;
      const result = fitRectToGrid(rect, gridSize);

      expect(result.x).toBe(25);
      expect(result.y).toBe(50);
      expect(result.width % gridSize).toBe(0);
      expect(result.height % gridSize).toBe(0);
      expect(result.width).toBeLessThanOrEqual(rect.width);
      expect(result.height).toBeLessThanOrEqual(rect.height);
    });

    it('should handle rect starting outside of grid alignment', () => {
      const rect: Rect = { x: 13, y: 27, width: 94, height: 112 };
      const gridSize = 20;
      const result = fitRectToGrid(rect, gridSize);

      expect(result.x).toBe(20);
      expect(result.y).toBe(40);
      expect(result.width % gridSize).toBe(0);
      expect(result.height % gridSize).toBe(0);
      expect(result.width).toBeLessThanOrEqual(rect.width);
      expect(result.height).toBeLessThanOrEqual(rect.height);
    });

    it('should return the same rect if already aligned to grid', () => {
      const rect: Rect = { x: 100, y: 100, width: 200, height: 300 };
      const gridSize = 50;
      const result = fitRectToGrid(rect, gridSize);

      expect(result).toEqual(rect);
    });

    it('should handle large grid sizes relative to rect dimensions', () => {
      const rect: Rect = { x: 250, y: 300, width: 400, height: 500 };
      const gridSize = 100;
      const result = fitRectToGrid(rect, gridSize);

      expect(result.x).toBe(300);
      expect(result.y).toBe(300);
      expect(result.width % gridSize).toBe(0);
      expect(result.height % gridSize).toBe(0);
      expect(result.width).toBeLessThanOrEqual(rect.width);
      expect(result.height).toBeLessThanOrEqual(rect.height);
    });

    it('should handle rect with zero width and height', () => {
      const rect: Rect = { x: 40, y: 60, width: 100, height: 200 };
      const gridSize = 20;
      const result = fitRectToGrid(rect, gridSize);

      expect(result).toEqual({ x: 40, y: 60, width: 100, height: 200 });
    });
  });
});

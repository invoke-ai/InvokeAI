import { getPrefixedId, getRectUnion } from 'features/controlLayers/konva/util';
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
});

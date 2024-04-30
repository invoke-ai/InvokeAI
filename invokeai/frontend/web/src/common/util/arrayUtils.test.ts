import { moveBackward, moveForward, moveToBack, moveToFront } from 'common/util/arrayUtils';
import { describe, expect, it } from 'vitest';

describe('Array Manipulation Functions', () => {
  const originalArray = ['a', 'b', 'c', 'd'];
  describe('moveForwardOne', () => {
    it('should move an item forward by one position', () => {
      const array = [...originalArray];
      const result = moveForward(array, (item) => item === 'b');
      expect(result).toEqual(['a', 'c', 'b', 'd']);
    });

    it('should do nothing if the item is at the end', () => {
      const array = [...originalArray];
      const result = moveForward(array, (item) => item === 'd');
      expect(result).toEqual(['a', 'b', 'c', 'd']);
    });

    it("should leave the array unchanged if the item isn't in the array", () => {
      const array = [...originalArray];
      const result = moveForward(array, (item) => item === 'z');
      expect(result).toEqual(originalArray);
    });
  });

  describe('moveToFront', () => {
    it('should move an item to the front', () => {
      const array = [...originalArray];
      const result = moveToFront(array, (item) => item === 'c');
      expect(result).toEqual(['c', 'a', 'b', 'd']);
    });

    it('should do nothing if the item is already at the front', () => {
      const array = [...originalArray];
      const result = moveToFront(array, (item) => item === 'a');
      expect(result).toEqual(['a', 'b', 'c', 'd']);
    });

    it("should leave the array unchanged if the item isn't in the array", () => {
      const array = [...originalArray];
      const result = moveToFront(array, (item) => item === 'z');
      expect(result).toEqual(originalArray);
    });
  });

  describe('moveBackwardsOne', () => {
    it('should move an item backward by one position', () => {
      const array = [...originalArray];
      const result = moveBackward(array, (item) => item === 'c');
      expect(result).toEqual(['a', 'c', 'b', 'd']);
    });

    it('should do nothing if the item is at the beginning', () => {
      const array = [...originalArray];
      const result = moveBackward(array, (item) => item === 'a');
      expect(result).toEqual(['a', 'b', 'c', 'd']);
    });

    it("should leave the array unchanged if the item isn't in the array", () => {
      const array = [...originalArray];
      const result = moveBackward(array, (item) => item === 'z');
      expect(result).toEqual(originalArray);
    });
  });

  describe('moveToBack', () => {
    it('should move an item to the back', () => {
      const array = [...originalArray];
      const result = moveToBack(array, (item) => item === 'b');
      expect(result).toEqual(['a', 'c', 'd', 'b']);
    });

    it('should do nothing if the item is already at the back', () => {
      const array = [...originalArray];
      const result = moveToBack(array, (item) => item === 'd');
      expect(result).toEqual(['a', 'b', 'c', 'd']);
    });

    it("should leave the array unchanged if the item isn't in the array", () => {
      const array = [...originalArray];
      const result = moveToBack(array, (item) => item === 'z');
      expect(result).toEqual(originalArray);
    });
  });
});

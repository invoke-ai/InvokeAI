import { moveOneToEnd, moveOneToStart, moveToEnd, moveToStart } from 'common/util/arrayUtils';
import { describe, expect, it } from 'vitest';

describe('Array Manipulation Functions', () => {
  const originalArray = ['a', 'b', 'c', 'd'];

  describe('moveOneToEnd', () => {
    describe('with callback', () => {
      it('should move an item forward by one position', () => {
        const array = [...originalArray];
        const result = moveOneToEnd(array, (item) => item === 'b');
        expect(result).toEqual(['a', 'c', 'b', 'd']);
      });

      it('should do nothing if the item is at the end', () => {
        const array = [...originalArray];
        const result = moveOneToEnd(array, (item) => item === 'd');
        expect(result).toEqual(['a', 'b', 'c', 'd']);
      });

      it("should leave the array unchanged if the item isn't in the array", () => {
        const array = [...originalArray];
        const result = moveOneToEnd(array, (item) => item === 'z');
        expect(result).toEqual(originalArray);
      });
    });
    describe('with item', () => {
      it('should move an item forward by one position', () => {
        const array = [...originalArray];
        const result = moveOneToEnd(array, (item) => item === 'b');
        expect(result).toEqual(['a', 'c', 'b', 'd']);
      });

      it('should do nothing if the item is at the end', () => {
        const array = [...originalArray];
        const result = moveOneToEnd(array, (item) => item === 'd');
        expect(result).toEqual(['a', 'b', 'c', 'd']);
      });

      it("should leave the array unchanged if the item isn't in the array", () => {
        const array = [...originalArray];
        const result = moveOneToEnd(array, (item) => item === 'z');
        expect(result).toEqual(originalArray);
      });
    });
  });

  describe('moveToStart', () => {
    describe('with callback', () => {
      it('should move an item to the front', () => {
        const array = [...originalArray];
        const result = moveToStart(array, (item) => item === 'c');
        expect(result).toEqual(['c', 'a', 'b', 'd']);
      });

      it('should do nothing if the item is already at the front', () => {
        const array = [...originalArray];
        const result = moveToStart(array, (item) => item === 'a');
        expect(result).toEqual(['a', 'b', 'c', 'd']);
      });

      it("should leave the array unchanged if the item isn't in the array", () => {
        const array = [...originalArray];
        const result = moveToStart(array, (item) => item === 'z');
        expect(result).toEqual(originalArray);
      });
    });
    describe('with item', () => {
      it('should move an item to the front', () => {
        const array = [...originalArray];
        const result = moveToStart(array, 'c');
        expect(result).toEqual(['c', 'a', 'b', 'd']);
      });

      it('should do nothing if the item is already at the front', () => {
        const array = [...originalArray];
        const result = moveToStart(array, 'a');
        expect(result).toEqual(['a', 'b', 'c', 'd']);
      });

      it("should leave the array unchanged if the item isn't in the array", () => {
        const array = [...originalArray];
        const result = moveToStart(array, 'z');
        expect(result).toEqual(originalArray);
      });
    });
  });

  describe('moveOneToStart', () => {
    describe('with callback', () => {
      it('should move an item backward by one position', () => {
        const array = [...originalArray];
        const result = moveOneToStart(array, (item) => item === 'c');
        expect(result).toEqual(['a', 'c', 'b', 'd']);
      });

      it('should do nothing if the item is at the beginning', () => {
        const array = [...originalArray];
        const result = moveOneToStart(array, (item) => item === 'a');
        expect(result).toEqual(['a', 'b', 'c', 'd']);
      });

      it("should leave the array unchanged if the item isn't in the array", () => {
        const array = [...originalArray];
        const result = moveOneToStart(array, (item) => item === 'z');
        expect(result).toEqual(originalArray);
      });
    });
    describe('with item', () => {
      it('should move an item backward by one position', () => {
        const array = [...originalArray];
        const result = moveOneToStart(array, 'c');
        expect(result).toEqual(['a', 'c', 'b', 'd']);
      });

      it('should do nothing if the item is at the beginning', () => {
        const array = [...originalArray];
        const result = moveOneToStart(array, 'a');
        expect(result).toEqual(['a', 'b', 'c', 'd']);
      });

      it("should leave the array unchanged if the item isn't in the array", () => {
        const array = [...originalArray];
        const result = moveOneToStart(array, 'z');
        expect(result).toEqual(originalArray);
      });
    });
  });

  describe('moveToEnd', () => {
    describe('with callback', () => {
      it('should move an item to the back', () => {
        const array = [...originalArray];
        const result = moveToEnd(array, (item) => item === 'b');
        expect(result).toEqual(['a', 'c', 'd', 'b']);
      });

      it('should do nothing if the item is already at the back', () => {
        const array = [...originalArray];
        const result = moveToEnd(array, (item) => item === 'd');
        expect(result).toEqual(['a', 'b', 'c', 'd']);
      });

      it("should leave the array unchanged if the item isn't in the array", () => {
        const array = [...originalArray];
        const result = moveToEnd(array, (item) => item === 'z');
        expect(result).toEqual(originalArray);
      });
    });
    describe('with item', () => {
      it('should move an item to the back', () => {
        const array = [...originalArray];
        const result = moveToEnd(array, 'b');
        expect(result).toEqual(['a', 'c', 'd', 'b']);
      });

      it('should do nothing if the item is already at the back', () => {
        const array = [...originalArray];
        const result = moveToEnd(array, 'd');
        expect(result).toEqual(['a', 'b', 'c', 'd']);
      });

      it("should leave the array unchanged if the item isn't in the array", () => {
        const array = [...originalArray];
        const result = moveToEnd(array, 'z');
        expect(result).toEqual(originalArray);
      });
    });
  });
});

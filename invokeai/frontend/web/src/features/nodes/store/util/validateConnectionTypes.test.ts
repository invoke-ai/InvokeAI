import { describe, expect, it } from 'vitest';

import { validateConnectionTypes } from './validateConnectionTypes';

describe(validateConnectionTypes.name, () => {
  describe('generic cases', () => {
    it('should accept Scalar to Scalar of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', isCollection: false, isCollectionOrScalar: false },
        { name: 'FooField', isCollection: false, isCollectionOrScalar: false }
      );
      expect(r).toBe(true);
    });
    it('should accept Collection to Collection of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', isCollection: true, isCollectionOrScalar: false },
        { name: 'FooField', isCollection: true, isCollectionOrScalar: false }
      );
      expect(r).toBe(true);
    });
    it('should accept Scalar to CollectionOrScalar of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', isCollection: false, isCollectionOrScalar: false },
        { name: 'FooField', isCollection: false, isCollectionOrScalar: true }
      );
      expect(r).toBe(true);
    });
    it('should accept Collection to CollectionOrScalar of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', isCollection: true, isCollectionOrScalar: false },
        { name: 'FooField', isCollection: false, isCollectionOrScalar: true }
      );
      expect(r).toBe(true);
    });
    it('should reject Collection to Scalar of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', isCollection: true, isCollectionOrScalar: false },
        { name: 'FooField', isCollection: false, isCollectionOrScalar: false }
      );
      expect(r).toBe(false);
    });
    it('should reject CollectionOrScalar to Scalar of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', isCollection: false, isCollectionOrScalar: true },
        { name: 'FooField', isCollection: false, isCollectionOrScalar: false }
      );
      expect(r).toBe(false);
    });
    it('should reject mismatched types', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', isCollection: false, isCollectionOrScalar: false },
        { name: 'BarField', isCollection: false, isCollectionOrScalar: false }
      );
      expect(r).toBe(false);
    });
  });

  describe('special cases', () => {
    it('should reject a collection input to a collection input', () => {
      const r = validateConnectionTypes(
        { name: 'CollectionField', isCollection: true, isCollectionOrScalar: false },
        { name: 'CollectionField', isCollection: true, isCollectionOrScalar: false }
      );
      expect(r).toBe(false);
    });

    it('should accept equal types', () => {
      const r = validateConnectionTypes(
        { name: 'IntegerField', isCollection: false, isCollectionOrScalar: false },
        { name: 'IntegerField', isCollection: false, isCollectionOrScalar: false }
      );
      expect(r).toBe(true);
    });

    describe('CollectionItemField', () => {
      it('should accept CollectionItemField to any Scalar target', () => {
        const r = validateConnectionTypes(
          { name: 'CollectionItemField', isCollection: false, isCollectionOrScalar: false },
          { name: 'IntegerField', isCollection: false, isCollectionOrScalar: false }
        );
        expect(r).toBe(true);
      });
      it('should accept CollectionItemField to any CollectionOrScalar target', () => {
        const r = validateConnectionTypes(
          { name: 'CollectionItemField', isCollection: false, isCollectionOrScalar: false },
          { name: 'IntegerField', isCollection: false, isCollectionOrScalar: true }
        );
        expect(r).toBe(true);
      });
      it('should accept any non-Collection to CollectionItemField', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', isCollection: false, isCollectionOrScalar: false },
          { name: 'CollectionItemField', isCollection: false, isCollectionOrScalar: false }
        );
        expect(r).toBe(true);
      });
      it('should reject any Collection to CollectionItemField', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', isCollection: true, isCollectionOrScalar: false },
          { name: 'CollectionItemField', isCollection: false, isCollectionOrScalar: false }
        );
        expect(r).toBe(false);
      });
      it('should reject any CollectionOrScalar to CollectionItemField', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', isCollection: false, isCollectionOrScalar: true },
          { name: 'CollectionItemField', isCollection: false, isCollectionOrScalar: false }
        );
        expect(r).toBe(false);
      });
    });

    describe('CollectionOrScalar', () => {
      it('should accept any Scalar of same type to CollectionOrScalar', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', isCollection: false, isCollectionOrScalar: false },
          { name: 'IntegerField', isCollection: false, isCollectionOrScalar: true }
        );
        expect(r).toBe(true);
      });
      it('should accept any Collection of same type to CollectionOrScalar', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', isCollection: true, isCollectionOrScalar: false },
          { name: 'IntegerField', isCollection: false, isCollectionOrScalar: true }
        );
        expect(r).toBe(true);
      });
      it('should accept any CollectionOrScalar of same type to CollectionOrScalar', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', isCollection: false, isCollectionOrScalar: true },
          { name: 'IntegerField', isCollection: false, isCollectionOrScalar: true }
        );
        expect(r).toBe(true);
      });
    });

    describe('CollectionField', () => {
      it('should accept any CollectionField to any Collection type', () => {
        const r = validateConnectionTypes(
          { name: 'CollectionField', isCollection: false, isCollectionOrScalar: false },
          { name: 'IntegerField', isCollection: true, isCollectionOrScalar: false }
        );
        expect(r).toBe(true);
      });
      it('should accept any CollectionField to any CollectionOrScalar type', () => {
        const r = validateConnectionTypes(
          { name: 'CollectionField', isCollection: false, isCollectionOrScalar: false },
          { name: 'IntegerField', isCollection: false, isCollectionOrScalar: true }
        );
        expect(r).toBe(true);
      });
    });

    describe('subtype handling', () => {
      type TypePair = { t1: string; t2: string };
      const typePairs = [
        { t1: 'IntegerField', t2: 'FloatField' },
        { t1: 'IntegerField', t2: 'StringField' },
        { t1: 'FloatField', t2: 'StringField' },
      ];
      it.each(typePairs)('should accept Scalar $t1 to Scalar $t2', ({ t1, t2 }: TypePair) => {
        const r = validateConnectionTypes(
          { name: t1, isCollection: false, isCollectionOrScalar: false },
          { name: t2, isCollection: false, isCollectionOrScalar: false }
        );
        expect(r).toBe(true);
      });
      it.each(typePairs)('should accept Scalar $t1 to CollectionOrScalar $t2', ({ t1, t2 }: TypePair) => {
        const r = validateConnectionTypes(
          { name: t1, isCollection: false, isCollectionOrScalar: false },
          { name: t2, isCollection: false, isCollectionOrScalar: true }
        );
        expect(r).toBe(true);
      });
      it.each(typePairs)('should accept Collection $t1 to Collection $t2', ({ t1, t2 }: TypePair) => {
        const r = validateConnectionTypes(
          { name: t1, isCollection: true, isCollectionOrScalar: false },
          { name: t2, isCollection: false, isCollectionOrScalar: false }
        );
        expect(r).toBe(true);
      });
      it.each(typePairs)('should accept Collection $t1 to CollectionOrScalar $t2', ({ t1, t2 }: TypePair) => {
        const r = validateConnectionTypes(
          { name: t1, isCollection: true, isCollectionOrScalar: false },
          { name: t2, isCollection: false, isCollectionOrScalar: true }
        );
        expect(r).toBe(true);
      });
      it.each(typePairs)('should accept CollectionOrScalar $t1 to CollectionOrScalar $t2', ({ t1, t2 }: TypePair) => {
        const r = validateConnectionTypes(
          { name: t1, isCollection: false, isCollectionOrScalar: true },
          { name: t2, isCollection: false, isCollectionOrScalar: true }
        );
        expect(r).toBe(true);
      });
    });

    describe('AnyField', () => {
      it('should accept any Scalar type  to AnyField', () => {
        const r = validateConnectionTypes(
          { name: 'FooField', isCollection: false, isCollectionOrScalar: false },
          { name: 'AnyField', isCollection: false, isCollectionOrScalar: false }
        );
        expect(r).toBe(true);
      });
      it('should accept any Collection type  to AnyField', () => {
        const r = validateConnectionTypes(
          { name: 'FooField', isCollection: false, isCollectionOrScalar: false },
          { name: 'AnyField', isCollection: true, isCollectionOrScalar: false }
        );
        expect(r).toBe(true);
      });
      it('should accept any CollectionOrScalar type  to AnyField', () => {
        const r = validateConnectionTypes(
          { name: 'FooField', isCollection: false, isCollectionOrScalar: false },
          { name: 'AnyField', isCollection: false, isCollectionOrScalar: true }
        );
        expect(r).toBe(true);
      });
    });
  });
});

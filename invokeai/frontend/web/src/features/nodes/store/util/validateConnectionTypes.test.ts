import { describe, expect, it } from 'vitest';

import { validateConnectionTypes } from './validateConnectionTypes';

describe(validateConnectionTypes.name, () => {
  describe('generic cases', () => {
    it('should accept SINGLE to SINGLE of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', cardinality: 'SINGLE' },
        { name: 'FooField', cardinality: 'SINGLE' }
      );
      expect(r).toBe(true);
    });
    it('should accept COLLECTION to COLLECTION of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', cardinality: 'COLLECTION' },
        { name: 'FooField', cardinality: 'COLLECTION' }
      );
      expect(r).toBe(true);
    });
    it('should accept SINGLE to SINGLE_OR_COLLECTION of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', cardinality: 'SINGLE' },
        { name: 'FooField', cardinality: 'SINGLE_OR_COLLECTION' }
      );
      expect(r).toBe(true);
    });
    it('should accept COLLECTION to SINGLE_OR_COLLECTION of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', cardinality: 'COLLECTION' },
        { name: 'FooField', cardinality: 'SINGLE_OR_COLLECTION' }
      );
      expect(r).toBe(true);
    });
    it('should reject COLLECTION to SINGLE of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', cardinality: 'COLLECTION' },
        { name: 'FooField', cardinality: 'SINGLE' }
      );
      expect(r).toBe(false);
    });
    it('should reject SINGLE_OR_COLLECTION to SINGLE of same type', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', cardinality: 'SINGLE_OR_COLLECTION' },
        { name: 'FooField', cardinality: 'SINGLE' }
      );
      expect(r).toBe(false);
    });
    it('should reject mismatched types', () => {
      const r = validateConnectionTypes(
        { name: 'FooField', cardinality: 'SINGLE' },
        { name: 'BarField', cardinality: 'SINGLE' }
      );
      expect(r).toBe(false);
    });
  });

  describe('special cases', () => {
    it('should reject a COLLECTION input to a COLLECTION input', () => {
      const r = validateConnectionTypes(
        { name: 'CollectionField', cardinality: 'COLLECTION' },
        { name: 'CollectionField', cardinality: 'COLLECTION' }
      );
      expect(r).toBe(false);
    });

    it('should accept equal types', () => {
      const r = validateConnectionTypes(
        { name: 'IntegerField', cardinality: 'SINGLE' },
        { name: 'IntegerField', cardinality: 'SINGLE' }
      );
      expect(r).toBe(true);
    });

    describe('CollectionItemField', () => {
      it('should accept CollectionItemField to any SINGLE target', () => {
        const r = validateConnectionTypes(
          { name: 'CollectionItemField', cardinality: 'SINGLE' },
          { name: 'IntegerField', cardinality: 'SINGLE' }
        );
        expect(r).toBe(true);
      });
      it('should accept CollectionItemField to any SINGLE_OR_COLLECTION target', () => {
        const r = validateConnectionTypes(
          { name: 'CollectionItemField', cardinality: 'SINGLE' },
          { name: 'IntegerField', cardinality: 'SINGLE_OR_COLLECTION' }
        );
        expect(r).toBe(true);
      });
      it('should accept any SINGLE to CollectionItemField', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', cardinality: 'SINGLE' },
          { name: 'CollectionItemField', cardinality: 'SINGLE' }
        );
        expect(r).toBe(true);
      });
      it('should reject any COLLECTION to CollectionItemField', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', cardinality: 'COLLECTION' },
          { name: 'CollectionItemField', cardinality: 'SINGLE' }
        );
        expect(r).toBe(false);
      });
      it('should reject any SINGLE_OR_COLLECTION to CollectionItemField', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', cardinality: 'SINGLE_OR_COLLECTION' },
          { name: 'CollectionItemField', cardinality: 'SINGLE' }
        );
        expect(r).toBe(false);
      });
    });

    describe('SINGLE_OR_COLLECTION', () => {
      it('should accept any SINGLE of same type to SINGLE_OR_COLLECTION', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', cardinality: 'SINGLE' },
          { name: 'IntegerField', cardinality: 'SINGLE_OR_COLLECTION' }
        );
        expect(r).toBe(true);
      });
      it('should accept any COLLECTION of same type to SINGLE_OR_COLLECTION', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', cardinality: 'COLLECTION' },
          { name: 'IntegerField', cardinality: 'SINGLE_OR_COLLECTION' }
        );
        expect(r).toBe(true);
      });
      it('should accept any SINGLE_OR_COLLECTION of same type to SINGLE_OR_COLLECTION', () => {
        const r = validateConnectionTypes(
          { name: 'IntegerField', cardinality: 'SINGLE_OR_COLLECTION' },
          { name: 'IntegerField', cardinality: 'SINGLE_OR_COLLECTION' }
        );
        expect(r).toBe(true);
      });
    });

    describe('CollectionField', () => {
      it('should accept any CollectionField to any COLLECTION type', () => {
        const r = validateConnectionTypes(
          { name: 'CollectionField', cardinality: 'SINGLE' },
          { name: 'IntegerField', cardinality: 'COLLECTION' }
        );
        expect(r).toBe(true);
      });
      it('should accept any CollectionField to any SINGLE_OR_COLLECTION type', () => {
        const r = validateConnectionTypes(
          { name: 'CollectionField', cardinality: 'SINGLE' },
          { name: 'IntegerField', cardinality: 'SINGLE_OR_COLLECTION' }
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
      it.each(typePairs)('should accept SINGLE $t1 to SINGLE $t2', ({ t1, t2 }: TypePair) => {
        const r = validateConnectionTypes({ name: t1, cardinality: 'SINGLE' }, { name: t2, cardinality: 'SINGLE' });
        expect(r).toBe(true);
      });
      it.each(typePairs)('should accept SINGLE $t1 to SINGLE_OR_COLLECTION $t2', ({ t1, t2 }: TypePair) => {
        const r = validateConnectionTypes(
          { name: t1, cardinality: 'SINGLE' },
          { name: t2, cardinality: 'SINGLE_OR_COLLECTION' }
        );
        expect(r).toBe(true);
      });
      it.each(typePairs)('should accept COLLECTION $t1 to COLLECTION $t2', ({ t1, t2 }: TypePair) => {
        const r = validateConnectionTypes(
          { name: t1, cardinality: 'COLLECTION' },
          { name: t2, cardinality: 'COLLECTION' }
        );
        expect(r).toBe(true);
      });
      it.each(typePairs)('should accept COLLECTION $t1 to SINGLE_OR_COLLECTION $t2', ({ t1, t2 }: TypePair) => {
        const r = validateConnectionTypes(
          { name: t1, cardinality: 'COLLECTION' },
          { name: t2, cardinality: 'SINGLE_OR_COLLECTION' }
        );
        expect(r).toBe(true);
      });
      it.each(typePairs)(
        'should accept SINGLE_OR_COLLECTION $t1 to SINGLE_OR_COLLECTION $t2',
        ({ t1, t2 }: TypePair) => {
          const r = validateConnectionTypes(
            { name: t1, cardinality: 'SINGLE_OR_COLLECTION' },
            { name: t2, cardinality: 'SINGLE_OR_COLLECTION' }
          );
          expect(r).toBe(true);
        }
      );
    });

    describe('AnyField', () => {
      it('should accept any SINGLE type  to AnyField', () => {
        const r = validateConnectionTypes(
          { name: 'FooField', cardinality: 'SINGLE' },
          { name: 'AnyField', cardinality: 'SINGLE' }
        );
        expect(r).toBe(true);
      });
      it('should accept any COLLECTION type  to AnyField', () => {
        const r = validateConnectionTypes(
          { name: 'FooField', cardinality: 'SINGLE' },
          { name: 'AnyField', cardinality: 'COLLECTION' }
        );
        expect(r).toBe(true);
      });
      it('should accept any SINGLE_OR_COLLECTION type  to AnyField', () => {
        const r = validateConnectionTypes(
          { name: 'FooField', cardinality: 'SINGLE' },
          { name: 'AnyField', cardinality: 'SINGLE_OR_COLLECTION' }
        );
        expect(r).toBe(true);
      });
    });
  });
});

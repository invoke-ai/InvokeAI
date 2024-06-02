import type { FieldType } from 'features/nodes/types/field';
import { describe, expect, it } from 'vitest';

import { areTypesEqual } from './areTypesEqual';

describe(areTypesEqual.name, () => {
  it('should handle equal source and target type', () => {
    const sourceType: FieldType = {
      name: 'IntegerField',
      cardinality: 'SINGLE',
      originalType: {
        name: 'Foo',
        cardinality: 'SINGLE',
      },
    };
    const targetType: FieldType = {
      name: 'IntegerField',
      cardinality: 'SINGLE',
      originalType: {
        name: 'Bar',
        cardinality: 'SINGLE',
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });

  it('should handle equal source type and original target type', () => {
    const sourceType: FieldType = {
      name: 'IntegerField',
      cardinality: 'SINGLE',
      originalType: {
        name: 'Foo',
        cardinality: 'SINGLE',
      },
    };
    const targetType: FieldType = {
      name: 'MainModelField',
      cardinality: 'SINGLE',
      originalType: {
        name: 'IntegerField',
        cardinality: 'SINGLE',
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });

  it('should handle equal original source type and target type', () => {
    const sourceType: FieldType = {
      name: 'MainModelField',
      cardinality: 'SINGLE',
      originalType: {
        name: 'IntegerField',
        cardinality: 'SINGLE',
      },
    };
    const targetType: FieldType = {
      name: 'IntegerField',
      cardinality: 'SINGLE',
      originalType: {
        name: 'Bar',
        cardinality: 'SINGLE',
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });

  it('should handle equal original source type and original target type', () => {
    const sourceType: FieldType = {
      name: 'MainModelField',
      cardinality: 'SINGLE',
      originalType: {
        name: 'IntegerField',
        cardinality: 'SINGLE',
      },
    };
    const targetType: FieldType = {
      name: 'LoRAModelField',
      cardinality: 'SINGLE',
      originalType: {
        name: 'IntegerField',
        cardinality: 'SINGLE',
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });
});

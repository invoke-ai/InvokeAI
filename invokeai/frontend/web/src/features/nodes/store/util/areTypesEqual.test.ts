import type { FieldType } from 'features/nodes/types/field';
import { describe, expect, it } from 'vitest';

import { areTypesEqual } from './areTypesEqual';

describe(areTypesEqual.name, () => {
  it('should handle equal source and target type', () => {
    const sourceType: FieldType = {
      name: 'IntegerField',
      cardinality: 'SINGLE',
      batch: false,
      originalType: {
        name: 'Foo',
        cardinality: 'SINGLE',
        batch: false,
      },
    };
    const targetType: FieldType = {
      name: 'IntegerField',
      cardinality: 'SINGLE',
      batch: false,
      originalType: {
        name: 'Bar',
        cardinality: 'SINGLE',
        batch: false,
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });

  it('should handle equal source type and original target type', () => {
    const sourceType: FieldType = {
      name: 'IntegerField',
      cardinality: 'SINGLE',
      batch: false,
      originalType: {
        name: 'Foo',
        cardinality: 'SINGLE',
        batch: false,
      },
    };
    const targetType: FieldType = {
      name: 'MainModelField',
      cardinality: 'SINGLE',
      batch: false,
      originalType: {
        name: 'IntegerField',
        cardinality: 'SINGLE',
        batch: false,
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });

  it('should handle equal original source type and target type', () => {
    const sourceType: FieldType = {
      name: 'MainModelField',
      cardinality: 'SINGLE',
      batch: false,
      originalType: {
        name: 'IntegerField',
        cardinality: 'SINGLE',
        batch: false,
      },
    };
    const targetType: FieldType = {
      name: 'IntegerField',
      cardinality: 'SINGLE',
      batch: false,
      originalType: {
        name: 'Bar',
        cardinality: 'SINGLE',
        batch: false,
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });

  it('should handle equal original source type and original target type', () => {
    const sourceType: FieldType = {
      name: 'MainModelField',
      cardinality: 'SINGLE',
      batch: false,
      originalType: {
        name: 'IntegerField',
        cardinality: 'SINGLE',
        batch: false,
      },
    };
    const targetType: FieldType = {
      name: 'LoRAModelField',
      cardinality: 'SINGLE',
      batch: false,
      originalType: {
        name: 'IntegerField',
        cardinality: 'SINGLE',
        batch: false,
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });
});

import { describe, expect, it } from 'vitest';

import { areTypesEqual } from './areTypesEqual';

describe(areTypesEqual.name, () => {
  it('should handle equal source and target type', () => {
    const sourceType = {
      name: 'IntegerField',
      isCollection: false,
      isCollectionOrScalar: false,
      originalType: {
        name: 'Foo',
        isCollection: false,
        isCollectionOrScalar: false,
      },
    };
    const targetType = {
      name: 'IntegerField',
      isCollection: false,
      isCollectionOrScalar: false,
      originalType: {
        name: 'Bar',
        isCollection: false,
        isCollectionOrScalar: false,
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });

  it('should handle equal source type and original target type', () => {
    const sourceType = {
      name: 'IntegerField',
      isCollection: false,
      isCollectionOrScalar: false,
      originalType: {
        name: 'Foo',
        isCollection: false,
        isCollectionOrScalar: false,
      },
    };
    const targetType = {
      name: 'Bar',
      isCollection: false,
      isCollectionOrScalar: false,
      originalType: {
        name: 'IntegerField',
        isCollection: false,
        isCollectionOrScalar: false,
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });

  it('should handle equal original source type and target type', () => {
    const sourceType = {
      name: 'Foo',
      isCollection: false,
      isCollectionOrScalar: false,
      originalType: {
        name: 'IntegerField',
        isCollection: false,
        isCollectionOrScalar: false,
      },
    };
    const targetType = {
      name: 'IntegerField',
      isCollection: false,
      isCollectionOrScalar: false,
      originalType: {
        name: 'Bar',
        isCollection: false,
        isCollectionOrScalar: false,
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });

  it('should handle equal original source type and original target type', () => {
    const sourceType = {
      name: 'Foo',
      isCollection: false,
      isCollectionOrScalar: false,
      originalType: {
        name: 'IntegerField',
        isCollection: false,
        isCollectionOrScalar: false,
      },
    };
    const targetType = {
      name: 'Bar',
      isCollection: false,
      isCollectionOrScalar: false,
      originalType: {
        name: 'IntegerField',
        isCollection: false,
        isCollectionOrScalar: false,
      },
    };
    expect(areTypesEqual(sourceType, targetType)).toBe(true);
  });
});

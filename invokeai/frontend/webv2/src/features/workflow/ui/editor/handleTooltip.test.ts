import type { FieldType } from '@features/workflow/contracts';

import { describe, expect, it } from 'vitest';

import { getHandleTypeTooltip } from './handleTooltip';

const fieldType = (overrides: Partial<FieldType> = {}): FieldType => ({
  batch: false,
  cardinality: 'SINGLE',
  name: 'ImageField',
  ...overrides,
});

describe('handleTooltip', () => {
  it('uses the resolved field type label for handle titles', () => {
    expect(getHandleTypeTooltip(fieldType())).toBe('Image');
    expect(getHandleTypeTooltip(fieldType({ cardinality: 'COLLECTION' }))).toBe('Image Collection');
    expect(getHandleTypeTooltip(fieldType({ batch: true }))).toBe('Image batch');
  });

  it('uses fallbacks for unresolved connector handles', () => {
    expect(getHandleTypeTooltip(null, 'Any input')).toBe('Any input');
  });
});

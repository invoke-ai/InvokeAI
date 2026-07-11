import { describe, expect, it } from 'vitest';

import { resolveBottomControlSlots } from './CanvasBottomControls';

describe('canvas bottom controls', () => {
  it.each([
    { externalLock: false, operationKind: null, expected: { operation: false, regular: true }, scenario: 'idle' },
    { externalLock: true, operationKind: null, expected: { operation: false, regular: false }, scenario: 'staging' },
    { externalLock: true, operationKind: null, expected: { operation: false, regular: false }, scenario: 'generation' },
    {
      externalLock: false,
      operationKind: 'filter' as const,
      expected: { operation: true, regular: false },
      scenario: 'active Filter',
    },
    {
      externalLock: true,
      operationKind: 'filter' as const,
      expected: { operation: true, regular: false },
      scenario: 'active Filter plus staging and temporary View',
    },
    {
      externalLock: false,
      operationKind: 'select-object' as const,
      expected: { operation: true, regular: false },
      scenario: 'active SAM',
    },
    {
      externalLock: true,
      operationKind: 'select-object' as const,
      expected: { operation: true, regular: false },
      scenario: 'active SAM plus generation and temporary View',
    },
  ])('resolves $scenario bottom controls', ({ expected, externalLock, operationKind }) => {
    expect(resolveBottomControlSlots({ isExternalInteractionLocked: externalLock, operationKind })).toEqual(expected);
  });
});

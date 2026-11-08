import { describe, expect, it } from 'vitest';

import { getProjectCanvasSchemaRequirement, isCanvasSchemaVersionSupported } from './canvasSchemaVersion';

describe('getProjectCanvasSchemaRequirement', () => {
  it('uses only the live canvas schema version', () => {
    expect(
      getProjectCanvasSchemaRequirement({
        canvas: { version: 2 },
        queue: { items: [{ snapshot: { canvas: { version: 99 } } }] },
      })
    ).toBe(2);
  });

  it.each([
    ['the live canvas is missing', {}],
    ['the live version is malformed', { canvas: { version: '2' } }],
  ])('fails closed when %s', (_label, document) => {
    expect(() => getProjectCanvasSchemaRequirement(document)).toThrow();
  });
});

describe('isCanvasSchemaVersionSupported', () => {
  it('opens projects needing schema 3 through 6, including control-layer derivation, and refuses the rest', () => {
    expect([2, 3, 4, 5, 6, 7].map(isCanvasSchemaVersionSupported)).toEqual([false, true, true, true, true, false]);
  });
});

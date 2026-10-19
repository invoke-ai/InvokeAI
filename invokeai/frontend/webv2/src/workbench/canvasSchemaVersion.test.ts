import { describe, expect, it } from 'vitest';

import { getProjectCanvasSchemaRequirement } from './canvasSchemaVersion';

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

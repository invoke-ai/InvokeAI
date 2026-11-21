import { describe, expect, it } from 'vitest';

import { getImageExportDimensions, normalizeImageExportSettings } from './imageExport';

describe('normalizeImageExportSettings', () => {
  it('reads a stored preference without the key as the built-in default', () => {
    expect(normalizeImageExportSettings(undefined)).toEqual({
      format: 'png',
      longEdge: null,
      metadata: 'all',
      namePattern: '{name}',
      quality: 90,
    });
  });

  it('keeps each usable field and replaces each malformed one on its own', () => {
    expect(
      normalizeImageExportSettings({
        format: 'webp',
        longEdge: 0,
        metadata: 'parameters',
        namePattern: '   ',
        quality: 140,
      })
    ).toEqual({ format: 'webp', longEdge: null, metadata: 'parameters', namePattern: '{name}', quality: 100 });
    expect(
      normalizeImageExportSettings({
        format: 'gif',
        longEdge: 1023.6,
        metadata: 'everything',
        namePattern: '{board}-{n}',
        quality: Number.NaN,
      })
    ).toEqual({ format: 'png', longEdge: 1024, metadata: 'all', namePattern: '{board}-{n}', quality: 90 });
    expect(normalizeImageExportSettings({ namePattern: 'x'.repeat(201), quality: -3 })).toMatchObject({
      namePattern: '{name}',
      quality: 1,
    });
  });
});

describe('getImageExportDimensions', () => {
  it('caps the longer side, keeps the aspect ratio, and never enlarges', () => {
    expect(getImageExportDimensions({ height: 768, width: 1024 }, 512)).toEqual({ height: 384, width: 512 });
    expect(getImageExportDimensions({ height: 1536, width: 640 }, 768)).toEqual({ height: 768, width: 320 });
    expect(getImageExportDimensions({ height: 768, width: 1024 }, 4096)).toEqual({ height: 768, width: 1024 });
    expect(getImageExportDimensions({ height: 768, width: 1024 }, null)).toEqual({ height: 768, width: 1024 });
  });

  it('rounds halves to even like the server and keeps each side at least one pixel', () => {
    // Python's round(2.5) is 2 and round(3.5) is 4.
    expect(getImageExportDimensions({ height: 5, width: 1000 }, 500)).toEqual({ height: 2, width: 500 });
    expect(getImageExportDimensions({ height: 7, width: 1000 }, 500)).toEqual({ height: 4, width: 500 });
    expect(getImageExportDimensions({ height: 10, width: 4000 }, 20)).toEqual({ height: 1, width: 20 });
  });
});

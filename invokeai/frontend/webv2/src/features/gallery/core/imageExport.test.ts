import { describe, expect, it } from 'vitest';

import { DEFAULT_IMAGE_EXPORT_SETTINGS, getImageExportDimensions, normalizeImageExportSettings } from './imageExport';

describe('image export settings', () => {
  it('normalizes an absent or malformed stored preference to usable settings', () => {
    expect(normalizeImageExportSettings(undefined)).toEqual(DEFAULT_IMAGE_EXPORT_SETTINGS);
    expect(
      normalizeImageExportSettings({ format: 'gif', longEdge: 1023.6, metadata: 1, namePattern: ' ', quality: 140 })
    ).toEqual({ format: 'png', longEdge: 1024, metadata: 'all', namePattern: '{name}', quality: 100 });
  });

  it('sizes like the server: caps the longer side, rounds halves to even, never enlarges', () => {
    expect(getImageExportDimensions({ height: 768, width: 1024 }, 512)).toEqual({ height: 384, width: 512 });
    expect(getImageExportDimensions({ height: 5, width: 1000 }, 500)).toEqual({ height: 2, width: 500 });
    expect(getImageExportDimensions({ height: 7, width: 1000 }, 500)).toEqual({ height: 4, width: 500 });
    expect(getImageExportDimensions({ height: 768, width: 1024 }, 4096)).toEqual({ height: 768, width: 1024 });
  });
});

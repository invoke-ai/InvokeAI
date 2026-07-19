import { describe, expect, it } from 'vitest';

import { shouldPublishGalleryTotal } from './GalleryWidgetView';

describe('shouldPublishGalleryTotal', () => {
  it('publishes finite totals that differ from known state and have not already been published', () => {
    expect(shouldPublishGalleryTotal({ knownTotalImages: null, lastPublishedTotal: null, total: 12 })).toBe(true);
    expect(shouldPublishGalleryTotal({ knownTotalImages: 8, lastPublishedTotal: null, total: 12 })).toBe(true);
  });

  it('does not republish the same in-flight total while known state catches up', () => {
    expect(shouldPublishGalleryTotal({ knownTotalImages: null, lastPublishedTotal: 12, total: 12 })).toBe(false);
  });

  it('does not publish non-finite or already-known totals', () => {
    expect(shouldPublishGalleryTotal({ knownTotalImages: 12, lastPublishedTotal: null, total: 12 })).toBe(false);
    expect(shouldPublishGalleryTotal({ knownTotalImages: null, lastPublishedTotal: null, total: null })).toBe(false);
    expect(shouldPublishGalleryTotal({ knownTotalImages: null, lastPublishedTotal: null, total: Number.NaN })).toBe(
      false
    );
  });
});

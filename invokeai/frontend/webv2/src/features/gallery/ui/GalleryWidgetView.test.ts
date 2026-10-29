import { formatGalleryMediaCounts } from '@features/gallery/core/mediaCounts';
import i18n from 'i18next';
import { describe, expect, it } from 'vitest';

import { shouldPublishGalleryTotal } from './GalleryWidgetView';

const englishCatalogModules = import.meta.glob('../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const englishCatalog = Object.values(englishCatalogModules)[0] as Record<string, unknown>;

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

describe('mixed-media gallery translations', () => {
  it('resolves the board, upload, status, and video labels with interpolation', async () => {
    const instance = i18n.createInstance();
    await instance.init({
      initAsync: false,
      lng: 'en',
      resources: { en: { translation: englishCatalog } },
    });

    expect(instance.t('widgets.gallery.statusChip', { count: 12 })).toBe('Gallery: 12 items');
    const t = (key: string, values?: Record<string, number | string>) => instance.t(key, values);
    expect(formatGalleryMediaCounts(t, { images: 2, uploads: 4, videos: 3 })).toBe('2 images, 3 videos, and 4 uploads');
    expect(instance.t('widgets.gallery.downloadBoardWithOmission', { count: 2 })).toBe(
      'Download board (2 videos omitted)'
    );
    expect(
      instance.t('widgets.gallery.uploadSummary', {
        board: 'Clips',
        media: formatGalleryMediaCounts(t, { images: 1, videos: 0 }),
      })
    ).toBe('1 image uploaded to Clips.');
    expect(instance.t('widgets.gallery.uploadFailedCount', { count: 1 })).toBe('1 file failed.');
    expect(instance.t('widgets.gallery.uploadSplit')).toBe('Images appear in Uploads; videos appear in Media.');
    expect(
      instance.t('widgets.gallery.deleteBoardMediaFailed', {
        media: formatGalleryMediaCounts(t, { images: 1, videos: 2 }),
      })
    ).toBe('1 image and 2 videos could not be deleted.');
    expect(instance.t('widgets.preview.videoLabel', { name: 'clip.mp4' })).toBe('Video clip.mp4');
    expect(instance.t('widgets.preview.videoFailed')).toBe('Video could not be loaded');
    expect(instance.t('widgets.preview.copyCurrentFrame')).toBe('Copy Current Frame');
    expect(instance.t('widgets.preview.videoDetails')).toBe('Video Details');
  });
});

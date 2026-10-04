import type { GalleryBoard } from '@features/gallery/core/types';
import type { TFunction } from 'i18next';

import { describe, expect, it } from 'vitest';

import { getGalleryUploadTargetLabel } from './GalleryUploadButton';

const t = ((key: string, values?: Record<string, unknown>) =>
  values && 'name' in values ? `${key}:${String(values.name)}` : key) as unknown as TFunction;

const board: GalleryBoard = {
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id: 'dogs',
  imageCount: 0,
  kind: 'board',
  name: 'Dogs',
  projectId: null,
  videoCount: 0,
};

describe('getGalleryUploadTargetLabel', () => {
  it('names the selected board', () => {
    expect(getGalleryUploadTargetLabel([board], 'dogs', t)).toEqual({
      isAvailable: true,
      label: 'widgets.gallery.uploadMediaToBoard:Dogs',
    });
  });

  it('falls back to a generic name for an unknown board', () => {
    expect(getGalleryUploadTargetLabel([board], 'missing', t)).toEqual({
      isAvailable: true,
      label: 'widgets.gallery.uploadMediaToBoard:widgets.gallery.selectedBoardFallback',
    });
  });

  it('marks date boards unavailable', () => {
    expect(getGalleryUploadTargetLabel([board], 'by_date:2026-09-04', t)).toEqual({
      isAvailable: false,
      label: 'widgets.gallery.uploadsUnavailableForDateBoards',
    });
  });
});

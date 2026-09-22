import { describe, expect, it } from 'vitest';

import { getMediaDeletionSummary } from './getMediaDeletionSummary';

describe('getMediaDeletionSummary', () => {
  it('reports a complete deletion as successful', () => {
    expect(getMediaDeletionSummary([{ status: 'fulfilled', value: { failed_images: [], failed_videos: [] } }])).toEqual(
      { failedCount: 0, requestFailed: false }
    );
  });

  it('counts image and video deletion failures', () => {
    expect(
      getMediaDeletionSummary([
        { status: 'fulfilled', value: { failed_images: ['image.png'], failed_videos: ['video.mp4'] } },
      ])
    ).toEqual({ failedCount: 2, requestFailed: false });
  });

  it('reports a rejected image deletion request', () => {
    expect(
      getMediaDeletionSummary([
        { status: 'rejected', reason: new Error('image delete failed') },
        { status: 'fulfilled', value: { failed_videos: [] } },
      ])
    ).toEqual({ failedCount: 0, requestFailed: true });
  });
});

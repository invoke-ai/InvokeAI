import { describe, expect, it } from 'vitest';

import { getMatchingProgressImage, getPreviewBoardsRequestKey } from './PreviewWidgetView';

describe('getPreviewBoardsRequestKey', () => {
  it('requests boards for local generated images, not only backend gallery images', () => {
    expect(
      getPreviewBoardsRequestKey({ hasSelectedImage: true, isBackendImage: false, refreshToken: 'refresh-1' })
    ).toBe('refresh-1');
  });

  it('does not request boards without a selected image', () => {
    expect(
      getPreviewBoardsRequestKey({ hasSelectedImage: false, isBackendImage: false, refreshToken: 'refresh-1' })
    ).toBeNull();
  });
});

describe('getMatchingProgressImage', () => {
  const placeholder = {
    boardId: 'none',
    height: 768,
    id: 'queue-1:1',
    itemIndex: 2,
    queueItemId: 'queue-1',
    width: 512,
  };
  const progressImage = {
    dataUrl: 'data:image/png;base64,abc',
    height: 768,
    target: { itemIndex: 2, queueItemId: 'queue-1' },
    width: 512,
  };

  it('returns progress only when it belongs to the current placeholder', () => {
    expect(getMatchingProgressImage(progressImage, placeholder)).toBe(progressImage);
    expect(
      getMatchingProgressImage({ ...progressImage, target: { itemIndex: 1, queueItemId: 'queue-1' } }, placeholder)
    ).toBeNull();
    expect(
      getMatchingProgressImage({ ...progressImage, target: { itemIndex: 2, queueItemId: 'queue-2' } }, placeholder)
    ).toBeNull();
  });
});

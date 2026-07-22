import { describe, expect, it } from 'vitest';

import { getMatchingProgressImage, getPreviewBoardsRequestKey, mergePreviewBoardImages } from './PreviewWidgetView';

describe('getPreviewBoardsRequestKey', () => {
  it('requests boards for local generated images, not only backend gallery images', () => {
    expect(getPreviewBoardsRequestKey({ hasSelectedImage: true, refreshToken: 'refresh-1' })).toBe('refresh-1');
  });

  it('does not request boards without a selected image', () => {
    expect(getPreviewBoardsRequestKey({ hasSelectedImage: false, refreshToken: 'refresh-1' })).toBeNull();
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

describe('mergePreviewBoardImages', () => {
  const image = (imageName: string, queuedAt: string, starred = false) => ({
    height: 64,
    imageName,
    imageUrl: `/images/${imageName}`,
    queuedAt,
    sourceQueueItemId: 'queue-1',
    starred,
    thumbnailUrl: `/images/${imageName}/thumbnail`,
    width: 64,
  });

  it('deduplicates and chronologically merges optimistic images in either direction', () => {
    const oldest = image('oldest', '2026-07-21T00:00:01.000Z');
    const middle = image('middle', '2026-07-21T00:00:02.000Z');
    const newest = image('newest', '2026-07-21T00:00:03.000Z');

    expect(mergePreviewBoardImages([newest, oldest], [middle, newest], 'DESC', false)).toEqual([
      newest,
      middle,
      oldest,
    ]);
    expect(mergePreviewBoardImages([oldest, newest], [middle, oldest], 'ASC', false)).toEqual([oldest, middle, newest]);
  });

  it('keeps starred backend images ahead of optimistic unstarred images', () => {
    const starred = image('starred', '2026-07-21T00:00:01.000Z', true);
    const optimistic = image('optimistic', '2026-07-21T00:00:03.000Z');
    const existing = image('existing', '2026-07-21T00:00:02.000Z');

    expect(mergePreviewBoardImages([starred, existing], [optimistic], 'DESC', true)).toEqual([
      starred,
      optimistic,
      existing,
    ]);
  });
});

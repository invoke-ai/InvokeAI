import { describe, expect, it } from 'vitest';

import { getDeletedImagesFromDeleteBoardAction } from './boardAndImagesDeleted';

describe('getDeletedImagesFromDeleteBoardAction', () => {
  it('reads deleted images from a successful response', () => {
    expect(getDeletedImagesFromDeleteBoardAction({ deleted_images: ['a.png'] })).toEqual(['a.png']);
  });

  it('reads deleted images from a partial-failure response', () => {
    expect(
      getDeletedImagesFromDeleteBoardAction({
        status: 500,
        data: { detail: { deleted_images: ['a.png'], deleted_videos: [], board_deleted: false } },
      })
    ).toEqual(['a.png']);
  });

  it('rejects malformed partial-failure data', () => {
    expect(getDeletedImagesFromDeleteBoardAction({ data: { detail: { deleted_images: [123] } } })).toEqual([]);
    expect(getDeletedImagesFromDeleteBoardAction(undefined)).toEqual([]);
  });
});

import { describe, expect, it } from 'vitest';

import { getPreviewBoardsRequestKey } from './PreviewWidgetView';

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

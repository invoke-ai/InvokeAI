import { getGalleryItemDragData } from '@features/gallery/utility';
import { describe, expect, it } from 'vitest';

import { PREVIEW_COMPARE_DROP_DATA, resolvePreviewCompareDrop } from './previewCompareDnd';

describe('resolvePreviewCompareDrop', () => {
  it('resolves the first dragged gallery image', () => {
    const activeData = getGalleryItemDragData([
      { kind: 'image', name: 'first.png' },
      { kind: 'image', name: 'second.png' },
    ]);

    expect(resolvePreviewCompareDrop(activeData, PREVIEW_COMPARE_DROP_DATA)).toEqual({ imageName: 'first.png' });
  });

  it('ignores drops that are not on the compare target', () => {
    const activeData = getGalleryItemDragData([{ kind: 'image', name: 'first.png' }]);

    expect(resolvePreviewCompareDrop(activeData, { kind: 'gallery-board' })).toBeNull();
    expect(resolvePreviewCompareDrop(activeData, null)).toBeNull();
  });

  it('ignores non-image, video, mixed, and empty drags', () => {
    expect(resolvePreviewCompareDrop({ kind: 'widget-instance' }, PREVIEW_COMPARE_DROP_DATA)).toBeNull();
    expect(
      resolvePreviewCompareDrop(
        getGalleryItemDragData([{ kind: 'video', name: 'clip.mp4' }]),
        PREVIEW_COMPARE_DROP_DATA
      )
    ).toBeNull();
    expect(
      resolvePreviewCompareDrop(
        getGalleryItemDragData([
          { kind: 'image', name: 'still.png' },
          { kind: 'video', name: 'clip.mp4' },
        ]),
        PREVIEW_COMPARE_DROP_DATA
      )
    ).toBeNull();
    expect(resolvePreviewCompareDrop({ items: [], kind: 'gallery-item' }, PREVIEW_COMPARE_DROP_DATA)).toBeNull();
  });
});

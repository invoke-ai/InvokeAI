import { getGalleryImageDragData } from '@workbench/widgets/gallery/galleryDnd';
import { describe, expect, it } from 'vitest';

import { PREVIEW_COMPARE_DROP_DATA, resolvePreviewCompareDrop } from './previewCompareDnd';

describe('resolvePreviewCompareDrop', () => {
  it('resolves the first dragged gallery image', () => {
    const activeData = getGalleryImageDragData([
      { boardId: 'none', imageName: 'first.png' },
      { boardId: 'none', imageName: 'second.png' },
    ]);

    expect(resolvePreviewCompareDrop(activeData, PREVIEW_COMPARE_DROP_DATA)).toEqual({ imageName: 'first.png' });
  });

  it('ignores drops that are not on the compare target', () => {
    const activeData = getGalleryImageDragData([{ boardId: 'none', imageName: 'first.png' }]);

    expect(resolvePreviewCompareDrop(activeData, { kind: 'gallery-board' })).toBeNull();
    expect(resolvePreviewCompareDrop(activeData, null)).toBeNull();
  });

  it('ignores non-image drags and empty drags', () => {
    expect(resolvePreviewCompareDrop({ kind: 'widget-instance' }, PREVIEW_COMPARE_DROP_DATA)).toBeNull();
    expect(resolvePreviewCompareDrop(getGalleryImageDragData([]), PREVIEW_COMPARE_DROP_DATA)).toBeNull();
  });
});

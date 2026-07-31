import { getGalleryItemDragData } from '@features/gallery/utility';
import { describe, expect, it } from 'vitest';

import { PREVIEW_COMPARE_DROP_DATA, resolvePreviewCompareDrop } from './previewCompareDnd';

/** Nothing on screen, so no drop can be a self-compare. */
const NOTHING_PREVIEWED = null;

describe('resolvePreviewCompareDrop', () => {
  it('resolves the first dragged gallery image', () => {
    const activeData = getGalleryItemDragData([
      { kind: 'image', name: 'first.png' },
      { kind: 'image', name: 'second.png' },
    ]);

    expect(resolvePreviewCompareDrop(activeData, PREVIEW_COMPARE_DROP_DATA, NOTHING_PREVIEWED)).toEqual({
      imageName: 'first.png',
    });
  });

  it('ignores drops that are not on the compare target', () => {
    const activeData = getGalleryItemDragData([{ kind: 'image', name: 'first.png' }]);

    expect(resolvePreviewCompareDrop(activeData, { kind: 'gallery-board' }, NOTHING_PREVIEWED)).toBeNull();
    expect(resolvePreviewCompareDrop(activeData, null, NOTHING_PREVIEWED)).toBeNull();
  });

  it('ignores non-image, video, mixed, and empty drags', () => {
    expect(
      resolvePreviewCompareDrop({ kind: 'widget-instance' }, PREVIEW_COMPARE_DROP_DATA, NOTHING_PREVIEWED)
    ).toBeNull();
    expect(
      resolvePreviewCompareDrop(
        getGalleryItemDragData([{ kind: 'video', name: 'clip.mp4' }]),
        PREVIEW_COMPARE_DROP_DATA,
        NOTHING_PREVIEWED
      )
    ).toBeNull();
    expect(
      resolvePreviewCompareDrop(
        getGalleryItemDragData([
          { kind: 'image', name: 'still.png' },
          { kind: 'video', name: 'clip.mp4' },
        ]),
        PREVIEW_COMPARE_DROP_DATA,
        NOTHING_PREVIEWED
      )
    ).toBeNull();
    expect(
      resolvePreviewCompareDrop({ items: [], kind: 'gallery-item' }, PREVIEW_COMPARE_DROP_DATA, NOTHING_PREVIEWED)
    ).toBeNull();
  });

  it('refuses to compare the previewed image with itself', () => {
    // Not merely a no-op: arming a comparison pauses live-follow, so this used
    // to switch off in-progress images and leave a comparison that sprang open
    // on the next selection.
    const activeData = getGalleryItemDragData([{ kind: 'image', name: 'shown.png' }]);

    expect(resolvePreviewCompareDrop(activeData, PREVIEW_COMPARE_DROP_DATA, 'shown.png')).toBeNull();
  });

  it('still resolves any other image while one is previewed', () => {
    const activeData = getGalleryItemDragData([{ kind: 'image', name: 'other.png' }]);

    expect(resolvePreviewCompareDrop(activeData, PREVIEW_COMPARE_DROP_DATA, 'shown.png')).toEqual({
      imageName: 'other.png',
    });
  });

  it('judges a multi-image drag by the image it would arm', () => {
    // Only `items[0]` is compared, because only `items[0]` is ever armed — a
    // drag whose *first* image is the previewed one is refused even though the
    // rest of the selection differs, and vice versa.
    const leadingIsPreviewed = getGalleryItemDragData([
      { kind: 'image', name: 'shown.png' },
      { kind: 'image', name: 'other.png' },
    ]);
    const trailingIsPreviewed = getGalleryItemDragData([
      { kind: 'image', name: 'other.png' },
      { kind: 'image', name: 'shown.png' },
    ]);

    expect(resolvePreviewCompareDrop(leadingIsPreviewed, PREVIEW_COMPARE_DROP_DATA, 'shown.png')).toBeNull();
    expect(resolvePreviewCompareDrop(trailingIsPreviewed, PREVIEW_COMPARE_DROP_DATA, 'shown.png')).toEqual({
      imageName: 'other.png',
    });
  });
});

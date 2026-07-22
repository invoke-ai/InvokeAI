import type { GalleryOrderDir, GalleryQueuePlaceholder, GalleryView } from '@features/gallery/contracts';

import { getGalleryPlaceholderInsertionIndex } from '@features/gallery/contracts';

/**
 * Pure navigation model for the preview widget's left/right stepping.
 *
 * The sequence is the current board's completed images plus, at most, the one
 * placeholder that is actively generating, placed where the gallery grid would
 * show it. Gallery selection and the live-follow preference remain the sources
 * of truth; the cursor is re-resolved from them on every render, never stored.
 */

interface NavigableImage {
  imageName: string;
  starred?: boolean;
}

export type PreviewNavigationItem<TImage extends NavigableImage> =
  | { kind: 'image'; image: TImage }
  | { kind: 'placeholder'; placeholder: GalleryQueuePlaceholder };

export const getPreviewNavigationSequence = <TImage extends NavigableImage>({
  activePlaceholder,
  boardId,
  boardImages,
  galleryView,
  imageOrderDir,
  starredFirst,
}: {
  /** The live slot from getGalleryGenerationSequence, or null. */
  activePlaceholder: GalleryQueuePlaceholder | null;
  /** The board backing boardImages — the selected image's own board. */
  boardId: string;
  /** Board images in the gallery's display order. */
  boardImages: TImage[];
  galleryView: GalleryView;
  imageOrderDir: GalleryOrderDir;
  starredFirst: boolean;
}): PreviewNavigationItem<TImage>[] => {
  const items: PreviewNavigationItem<TImage>[] = boardImages.map((image) => ({ image, kind: 'image' }));
  const includePlaceholder =
    activePlaceholder !== null && galleryView === 'images' && activePlaceholder.boardId === boardId;

  if (!includePlaceholder) {
    return items;
  }

  items.splice(getGalleryPlaceholderInsertionIndex(boardImages, imageOrderDir, starredFirst), 0, {
    kind: 'placeholder',
    placeholder: activePlaceholder,
  });

  return items;
};

export const getPreviewNavigationCursor = <TImage extends NavigableImage>(
  sequence: PreviewNavigationItem<TImage>[],
  { isFollowingLive, selectedImageName }: { isFollowingLive: boolean; selectedImageName: string | null }
): number => {
  if (isFollowingLive) {
    return sequence.findIndex((item) => item.kind === 'placeholder');
  }

  if (selectedImageName === null) {
    return -1;
  }

  return sequence.findIndex((item) => item.kind === 'image' && item.image.imageName === selectedImageName);
};

export const getPreviewNavigationTarget = <TImage extends NavigableImage>(
  sequence: PreviewNavigationItem<TImage>[],
  cursorIndex: number,
  offset: -1 | 1
): PreviewNavigationItem<TImage> | null => (cursorIndex === -1 ? null : (sequence[cursorIndex + offset] ?? null));

import type {
  GalleryItemKind,
  GalleryItemKey,
  GalleryOrderDir,
  GalleryQueuePlaceholder,
  GalleryView,
} from '@features/gallery/contracts';

import { getGalleryPlaceholderInsertionIndex, toGalleryItemKey } from '@features/gallery/contracts';

/**
 * Pure navigation model for the preview widget's left/right stepping.
 *
 * The sequence is the current board's completed images plus, at most, the one
 * placeholder that is actively generating, placed where the gallery grid would
 * show it. Gallery selection and the live-follow preference remain the sources
 * of truth; the cursor is re-resolved from them on every render, never stored.
 */

interface NavigableItem {
  kind: GalleryItemKind;
  name: string;
  starred?: boolean;
}

export type PreviewNavigationItem<TItem extends NavigableItem> =
  | { kind: 'item'; item: TItem }
  | { kind: 'placeholder'; placeholder: GalleryQueuePlaceholder };

export const getPreviewNavigationSequence = <TItem extends NavigableItem>({
  activePlaceholder,
  boardId,
  boardImages,
  galleryView,
  imageOrderDir,
  starredFirst,
}: {
  /** The live slot from getGalleryGenerationSequence, or null. */
  activePlaceholder: GalleryQueuePlaceholder | null;
  /** The board backing boardImages — the selected item's own board. */
  boardId: string;
  /** Board items in the gallery's display order. */
  boardImages: TItem[];
  galleryView: GalleryView;
  imageOrderDir: GalleryOrderDir;
  starredFirst: boolean;
}): PreviewNavigationItem<TItem>[] => {
  const items: PreviewNavigationItem<TItem>[] = boardImages.map((item) => ({ item, kind: 'item' }));
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

export const getPreviewNavigationCursor = <TItem extends NavigableItem>(
  sequence: PreviewNavigationItem<TItem>[],
  { isFollowingLive, selectedItemKey }: { isFollowingLive: boolean; selectedItemKey: GalleryItemKey | null }
): number => {
  if (isFollowingLive) {
    return sequence.findIndex((item) => item.kind === 'placeholder');
  }

  if (selectedItemKey === null) {
    return -1;
  }

  return sequence.findIndex((entry) => entry.kind === 'item' && toGalleryItemKey(entry.item) === selectedItemKey);
};

export const getPreviewNavigationTarget = <TItem extends NavigableItem>(
  sequence: PreviewNavigationItem<TItem>[],
  cursorIndex: number,
  offset: -1 | 1
): PreviewNavigationItem<TItem> | null => (cursorIndex === -1 ? null : (sequence[cursorIndex + offset] ?? null));

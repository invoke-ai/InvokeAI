import type { GalleryView } from '@features/gallery/core/types';

import { SegmentGroup, Text } from '@chakra-ui/react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { getGalleryCountForView } from './galleryBoardLabels';
import { useGalleryWidget } from './GalleryWidgetContext';

const GALLERY_VIEW_TABS = [
  { labelKey: 'common.media', value: 'images' },
  { labelKey: 'common.assets', value: 'assets' },
] satisfies { labelKey: string; value: GalleryView }[];
const CHECKED_VIEW_TAB_STYLES = { bg: 'accent.solid', color: 'accent.contrast' } as const;

/**
 * Media / Assets, each carrying the selected board's count for that view so
 * the split is legible before you switch.
 *
 * A segmented control rather than a tablist: this chooses which items the grid
 * queries and owns no panel of its own. Real tabs publish `aria-controls`
 * pointing at a tabpanel, and there is none to point at — the grid is a
 * sibling slot each layout shell places independently.
 */
export const GalleryViewTabs = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId);

  const handleViewChange = useCallback(
    (event: { value: string | null }) => {
      if (event.value) {
        actions.setView(event.value as GalleryView);
      }
    },
    [actions]
  );

  return (
    <SegmentGroup.Root
      aria-label={t('common.view')}
      // `2xs` is a repo recipe extension the generated Chakra types don't know.
      size={'2xs' as 'xs'}
      value={gallery.galleryView}
      onValueChange={handleViewChange}
    >
      <SegmentGroup.Indicator />
      {GALLERY_VIEW_TABS.map(({ labelKey, value }) => {
        const count = selectedBoard ? getGalleryCountForView(selectedBoard, value) : null;

        return (
          <SegmentGroup.Item key={value} value={value} _checked={CHECKED_VIEW_TAB_STYLES}>
            <SegmentGroup.ItemHiddenInput />
            <SegmentGroup.ItemText display="flex" fontSize="xs" gap="1.5">
              {t(labelKey)}
              {count === null ? null : (
                // Dimmed from the item's own text colour rather than pinned to
                // `fg.muted`: the checked item swaps to `accent.contrast`, and a
                // fixed muted grey is unreadable on the accent fill. 0.8 is the
                // dimmest that still clears 4.5:1 in both states.
                <Text as="span" color="currentColor" fontVariantNumeric="tabular-nums" opacity="0.8">
                  {count}
                </Text>
              )}
            </SegmentGroup.ItemText>
          </SegmentGroup.Item>
        );
      })}
    </SegmentGroup.Root>
  );
};

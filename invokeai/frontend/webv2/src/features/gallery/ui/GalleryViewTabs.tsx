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
      size="sm"
      value={gallery.galleryView}
      onValueChange={handleViewChange}
    >
      <SegmentGroup.Indicator />
      {GALLERY_VIEW_TABS.map(({ labelKey, value }) => {
        const count = selectedBoard ? getGalleryCountForView(selectedBoard, value) : null;

        return (
          <SegmentGroup.Item key={value} value={value}>
            <SegmentGroup.ItemHiddenInput />
            <SegmentGroup.ItemText display="flex" fontSize="xs" gap="1.5">
              {t(labelKey)}
              {count === null ? null : (
                <Text as="span" color="fg.muted" fontVariantNumeric="tabular-nums">
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

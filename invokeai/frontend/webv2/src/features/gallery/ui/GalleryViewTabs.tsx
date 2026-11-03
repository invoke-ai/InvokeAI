import type { GalleryBoard, GalleryView } from '@features/gallery/core/types';

import { Icon, Text } from '@chakra-ui/react';
import { SegmentTabs } from '@platform/ui';
import { ImagesIcon, ImageUpIcon, type LucideIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { getGalleryCountForView } from './galleryBoardLabels';
import { useGalleryWidget } from './GalleryWidgetContext';

const GALLERY_VIEW_TABS = [
  { icon: ImagesIcon, labelKey: 'common.media', value: 'images' },
  { icon: ImageUpIcon, labelKey: 'common.assets', value: 'assets' },
] satisfies { icon: LucideIcon; labelKey: string; value: GalleryView }[];

/** Wire the caller's grid to segmentTabsPanelId(idBase); tab counts distinguish Media and Assets before switching. */
export const GalleryViewSegmentTabs = ({
  activeView,
  board,
  iconLabels = false,
  idBase,
  onSelect,
}: {
  activeView: GalleryView;
  board: GalleryBoard | undefined;
  /** Icons in place of the view names, for headers too narrow to spare the words; the names become tooltips. */
  iconLabels?: boolean;
  idBase: string;
  onSelect: (view: GalleryView) => void;
}) => {
  const { t } = useTranslation();

  // SegmentTabs re-fires selecting the active tab (its collapsible-toggle
  // affordance); a same-view write would only dirty the widget values.
  const handleViewChange = useCallback(
    (value: GalleryView) => {
      if (value !== activeView) {
        onSelect(value);
      }
    },
    [activeView, onSelect]
  );

  const tabs = useMemo(
    () =>
      GALLERY_VIEW_TABS.map(({ icon, labelKey, value }) => {
        const count = board ? getGalleryCountForView(board, value) : null;
        const name = t(labelKey);

        return {
          ariaLabel: iconLabels ? (count === null ? name : `${name} ${count}`) : undefined,
          id: value,
          label: (
            <Text alignItems="center" as="span" display="flex" gap="1.5">
              {iconLabels ? <Icon as={icon} aria-hidden boxSize="3.5" /> : name}
              {count === null ? null : (
                <Text as="span" color="currentColor" fontVariantNumeric="tabular-nums" opacity="0.8">
                  {count}
                </Text>
              )}
            </Text>
          ),
        };
      }),
    [board, iconLabels, t]
  );

  return (
    <SegmentTabs
      activeId={activeView}
      ariaLabel={t('common.view')}
      idBase={idBase}
      isCompact
      tabs={tabs}
      onSelect={handleViewChange}
    />
  );
};

export const GalleryViewTabs = ({ idBase }: { idBase: string }) => {
  const { actions, gallery } = useGalleryWidget();
  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId);

  return (
    <GalleryViewSegmentTabs
      activeView={gallery.galleryView}
      board={selectedBoard}
      idBase={idBase}
      onSelect={actions.setView}
    />
  );
};

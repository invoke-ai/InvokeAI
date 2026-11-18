import type { GalleryMediaFilter } from '@features/gallery/core/settings';
import type { GalleryOrderDir } from '@features/gallery/core/types';
import type { RefObject } from 'react';

import { Menu } from '@chakra-ui/react';
import { DEFAULT_GALLERY_SETTINGS } from '@features/gallery/core/settings';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { GalleryFilterMenu } from './GalleryFilterMenu';
import { useGalleryWidget } from './GalleryWidgetContext';

const MEDIA_FILTER_LABEL_KEYS = {
  all: 'widgets.gallery.mediaFilterAll',
  image: 'widgets.gallery.mediaFilterImages',
  video: 'widgets.gallery.mediaFilterVideos',
} as const satisfies Record<GalleryMediaFilter, string>;

const ORDER_LABEL_KEYS = {
  ASC: 'widgets.gallery.oldest',
  DESC: 'widgets.gallery.newest',
} as const satisfies Record<GalleryOrderDir, string>;

/** Which media kinds the listing shows and in which order. The starred filter stays its own toggle. */
export const GalleryItemFilterMenu = ({
  triggerRef,
}: {
  /** The grid's "Show all media" hands focus here when it removes itself. */
  triggerRef: RefObject<HTMLButtonElement | null>;
}) => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const { imageOrderDir, mediaFilter } = gallery.settings;

  const handleMediaFilterChange = useCallback(
    (event: { value: string }) => actions.updateSettings({ mediaFilter: event.value as GalleryMediaFilter }),
    [actions]
  );

  const handleOrderDirChange = useCallback(
    (event: { value: string }) => actions.updateSettings({ imageOrderDir: event.value as GalleryOrderDir }),
    [actions]
  );

  return (
    <GalleryFilterMenu
      isFiltered={mediaFilter !== DEFAULT_GALLERY_SETTINGS.mediaFilter}
      label={t('widgets.gallery.filterAndSortMedia')}
      // The trigger shows only an icon, so its name states the current choices.
      triggerLabel={t('widgets.gallery.filterAndSortMediaValue', {
        media: t(MEDIA_FILTER_LABEL_KEYS[mediaFilter]),
        order: t(ORDER_LABEL_KEYS[imageOrderDir]),
      })}
      triggerRef={triggerRef}
    >
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('widgets.gallery.showMedia')}</Menu.ItemGroupLabel>
        <Menu.RadioItemGroup value={mediaFilter} onValueChange={handleMediaFilterChange}>
          {(['all', 'image', 'video'] as const).map((value) => (
            <Menu.RadioItem key={value} value={value}>
              <Menu.ItemIndicator />
              <Menu.ItemText>{t(MEDIA_FILTER_LABEL_KEYS[value])}</Menu.ItemText>
            </Menu.RadioItem>
          ))}
        </Menu.RadioItemGroup>
      </Menu.ItemGroup>
      <Menu.Separator />
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('widgets.gallery.sortBy')}</Menu.ItemGroupLabel>
        <Menu.RadioItemGroup value={imageOrderDir} onValueChange={handleOrderDirChange}>
          {(['DESC', 'ASC'] as const).map((value) => (
            <Menu.RadioItem key={value} value={value}>
              <Menu.ItemIndicator />
              <Menu.ItemText>{t(ORDER_LABEL_KEYS[value])}</Menu.ItemText>
            </Menu.RadioItem>
          ))}
        </Menu.RadioItemGroup>
      </Menu.ItemGroup>
    </GalleryFilterMenu>
  );
};

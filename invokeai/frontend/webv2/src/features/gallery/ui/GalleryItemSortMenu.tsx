import type { GalleryOrderDir } from '@features/gallery/core/types';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { ChevronDownIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { useGalleryWidget } from './GalleryWidgetContext';

const SORT_POSITIONING = { placement: 'bottom-end' } as const;

/**
 * Grid ordering, promoted out of the settings popover: which end of the board
 * you are looking at is a decision made constantly, not a preference set once.
 */
export const GalleryItemSortMenu = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const { imageOrderDir, starredFirst } = gallery.settings;

  const handleOrderDirChange = useCallback(
    (event: { value: string }) => actions.updateSettings({ imageOrderDir: event.value as GalleryOrderDir }),
    [actions]
  );

  const handleStarredFirstChange = useCallback(
    (starredFirst: boolean) => actions.updateSettings({ starredFirst }),
    [actions]
  );

  return (
    <Menu.Root positioning={SORT_POSITIONING}>
      <Menu.Trigger asChild>
        <Button
          aria-label={t('widgets.gallery.imageSort')}
          color="fg.muted"
          flexShrink={0}
          fontSize="xs"
          gap="1"
          size="xs"
          variant="ghost"
        >
          {imageOrderDir === 'DESC' ? t('widgets.gallery.newest') : t('widgets.gallery.oldest')}
          <Icon as={ChevronDownIcon} boxSize="3" />
        </Button>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="11rem">
            <Menu.RadioItemGroup value={imageOrderDir} onValueChange={handleOrderDirChange}>
              <Menu.RadioItem value="DESC">
                <Menu.ItemIndicator />
                <Menu.ItemText>{t('widgets.gallery.newest')}</Menu.ItemText>
              </Menu.RadioItem>
              <Menu.RadioItem value="ASC">
                <Menu.ItemIndicator />
                <Menu.ItemText>{t('widgets.gallery.oldest')}</Menu.ItemText>
              </Menu.RadioItem>
            </Menu.RadioItemGroup>
            <Menu.Separator />
            {/* A real checkbox item rather than a Switch smuggled into a menu
                item, which needed closeOnSelect + a click-swallower to behave. */}
            <Menu.CheckboxItem
              checked={starredFirst}
              closeOnSelect={false}
              value="starred-first"
              onCheckedChange={handleStarredFirstChange}
            >
              <Menu.ItemIndicator />
              <Menu.ItemText>{t('widgets.gallery.showStarredImagesFirst')}</Menu.ItemText>
            </Menu.CheckboxItem>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

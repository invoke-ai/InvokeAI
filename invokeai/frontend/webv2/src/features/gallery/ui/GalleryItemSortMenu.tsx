import type { GalleryOrderDir } from '@features/gallery/core/types';

import { Icon, Menu, Portal } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { ChevronDownIcon } from 'lucide-react';
import { useCallback, useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useGalleryWidget } from './GalleryWidgetContext';

const SORT_POSITIONING = { placement: 'bottom-end' } as const;

export const GalleryItemSortMenu = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const { imageOrderDir } = gallery.settings;
  // Shared ids let the tooltip ride the menu trigger without wrapping it
  // (wrapping `Menu.Trigger` swallows the anchor ref — see RoutingControl).
  const triggerId = useId();
  const triggerIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);

  const handleOrderDirChange = useCallback(
    (event: { value: string }) => actions.updateSettings({ imageOrderDir: event.value as GalleryOrderDir }),
    [actions]
  );

  return (
    <Menu.Root ids={triggerIds} positioning={SORT_POSITIONING}>
      <Tooltip content={t('widgets.gallery.sortBy')} ids={triggerIds}>
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
      </Tooltip>
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
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

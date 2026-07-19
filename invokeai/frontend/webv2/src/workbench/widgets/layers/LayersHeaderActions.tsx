import type { LucideIcon } from 'lucide-react';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { IconButton, MenuContent } from '@platform/ui';
import {
  BrushIcon,
  ImagePlusIcon,
  MapPinIcon,
  PlusIcon,
  SlidersHorizontalIcon,
  SquareDashedBottomIcon,
} from 'lucide-react';
import { Fragment, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { AddLayerItemId } from './addLayerMenu';

import { ADD_LAYER_MENU } from './addLayerMenu';
import { useAddLayer } from './useAddLayer';

const MENU_POSITIONING = { placement: 'bottom-end' } as const;

/** Icon per add-layer item (kept in the view; the menu structure itself is pure data). */
const ADD_LAYER_ICONS: Record<AddLayerItemId, LucideIcon> = {
  control: SlidersHorizontalIcon,
  inpaint_mask: SquareDashedBottomIcon,
  raster: BrushIcon,
  regional_guidance: MapPinIcon,
  regional_reference_image: ImagePlusIcon,
};

/**
 * Layers-panel header: the add-layer menu, split into legacy's two labelled groups
 * — "Regional" (inpaint mask / regional guidance / regional guidance + reference
 * image) and "Layers" (control / raster). The per-item creation is delegated to the
 * shared `useAddLayer` hook so this menu and each group header's "New" button agree.
 */
export const LayersHeaderActions = () => {
  const { t } = useTranslation();
  const addLayer = useAddLayer();

  const handleSelect = useCallback((id: AddLayerItemId) => () => addLayer(id), [addLayer]);

  return (
    <Menu.Root positioning={MENU_POSITIONING}>
      <Menu.Trigger asChild>
        <IconButton aria-label={t('widgets.layers.addLayer')} color="fg.muted" size="2xs" variant="ghost">
          <PlusIcon />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="12rem">
            {ADD_LAYER_MENU.map((group, groupIndex) => (
              <Fragment key={group.titleKey}>
                {groupIndex > 0 ? <Menu.Separator borderColor="border.subtle" /> : null}
                <Menu.ItemGroup>
                  <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                    {t(group.titleKey)}
                  </Menu.ItemGroupLabel>
                  {group.items.map((item) => {
                    const ItemIcon = ADD_LAYER_ICONS[item.id];
                    return (
                      <Menu.Item key={item.id} value={item.id} onSelect={handleSelect(item.id)}>
                        <HStack gap="2" minW="0" w="full">
                          <Icon as={ItemIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
                          <Text flex="1" fontSize="xs">
                            {t(item.labelKey)}
                          </Text>
                        </HStack>
                      </Menu.Item>
                    );
                  })}
                </Menu.ItemGroup>
              </Fragment>
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

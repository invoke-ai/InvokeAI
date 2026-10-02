import type { LucideIcon } from 'lucide-react';
import type { ComponentProps } from 'react';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { IconButton, MenuContent } from '@platform/ui';
import {
  BrushIcon,
  FolderPlusIcon,
  ImagePlusIcon,
  MapPinIcon,
  PlusIcon,
  SlidersHorizontalIcon,
  SquareDashedBottomIcon,
} from 'lucide-react';
import { Fragment, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { AddLayerItemId } from './addLayerMenu';
import type { LayerSurfaceAnchor } from './layerRowCommands';

import { ADD_LAYER_MENU, isAddLayerItemAvailable } from './addLayerMenu';
import { useAddLayer } from './useAddLayer';
import { useSelectedModelBase } from './useSelectedModelBase';

type MenuPositioning = ComponentProps<typeof Menu.Root>['positioning'];

const MENU_POSITIONING = { placement: 'bottom-end' } as const;

/** Icon per add-layer item (kept in the view; the menu structure itself is pure data). */
const ADD_LAYER_ICONS: Record<AddLayerItemId, LucideIcon> = {
  control: SlidersHorizontalIcon,
  group: FolderPlusIcon,
  inpaint_mask: SquareDashedBottomIcon,
  raster: BrushIcon,
  regional_guidance: MapPinIcon,
  regional_reference_image: ImagePlusIcon,
};

/** Share grouped add-layer choices between header and empty-area menus; useAddLayer owns creation. */
const AddLayerMenuItems = () => {
  const { t } = useTranslation();
  const addLayer = useAddLayer();
  const base = useSelectedModelBase();

  const handleSelect = useCallback((id: AddLayerItemId) => () => addLayer(id), [addLayer]);

  return (
    <MenuContent minW="12rem">
      {ADD_LAYER_MENU.map((group, groupIndex) => (
        <Fragment key={group.titleKey}>
          {groupIndex > 0 ? <Menu.Separator borderColor="border.subtle" /> : null}
          <Menu.ItemGroup>
            <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
              {t(group.titleKey)}
            </Menu.ItemGroupLabel>
            {group.items
              .filter((item) => isAddLayerItemAvailable(item.id, base))
              .map((item) => {
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
  );
};

/** The tree strip's "+": the add-layer menu behind a trigger button. */
export const LayersHeaderActions = () => {
  const { t } = useTranslation();

  return (
    <Menu.Root positioning={MENU_POSITIONING}>
      <Menu.Trigger asChild>
        <IconButton aria-label={t('widgets.layers.addLayer')} color="fg.muted" size="2xs" variant="ghost">
          <PlusIcon />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <AddLayerMenuItems />
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

/** The same add-layer menu opened at a right-click in the tree's empty space. */
export const AddLayerContextMenu = ({ anchor, onClose }: { anchor: LayerSurfaceAnchor; onClose: () => void }) => {
  const positioning = useMemo<MenuPositioning>(
    () => ({ getAnchorRect: () => anchor, placement: 'bottom-start' }),
    [anchor]
  );
  const handleOpenChange = useCallback(
    (details: { open: boolean }) => {
      if (!details.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Menu.Root lazyMount open positioning={positioning} unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Menu.Positioner>
          <AddLayerMenuItems />
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

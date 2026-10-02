import type { CreateFromBboxDestination } from '@workbench/canvas-operations/api';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { MenuContent } from '@platform/ui';
import { ChevronRightIcon, ImagePlusIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const SUBMENU_POSITIONING = { placement: 'right-start' } as const;

const CREATE_FROM_BBOX_ITEMS: readonly { destination: CreateFromBboxDestination; labelKey: string }[] = [
  { destination: 'global-reference', labelKey: 'widgets.canvas.contextMenu.newGlobalReferenceImage' },
  { destination: 'regional-reference', labelKey: 'widgets.canvas.contextMenu.newRegionalReferenceImage' },
  { destination: 'control', labelKey: 'widgets.canvas.contextMenu.newControlLayer' },
  { destination: 'raster', labelKey: 'widgets.canvas.contextMenu.newRasterLayer' },
];

const CreateFromBboxMenuItem = ({
  destination,
  disabled,
  labelKey,
  onCreate,
}: {
  destination: CreateFromBboxDestination;
  disabled: boolean;
  labelKey: string;
  onCreate: (destination: CreateFromBboxDestination) => void;
}) => {
  const { t } = useTranslation();
  const onSelect = useCallback(() => onCreate(destination), [destination, onCreate]);

  return (
    <Menu.Item disabled={disabled} value={destination} onSelect={onSelect}>
      <HStack gap="2" minW="0" w="full">
        <Icon as={ImagePlusIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
        <Text flex="1" fontSize="xs">
          {t(labelKey)}
        </Text>
      </HStack>
    </Menu.Item>
  );
};

export const CanvasCreateFromBboxSubmenu = ({
  disabled,
  onCreate,
}: {
  disabled: boolean;
  onCreate: (destination: CreateFromBboxDestination) => void;
}) => {
  const { t } = useTranslation();

  return (
    <Menu.Root positioning={SUBMENU_POSITIONING}>
      <Menu.TriggerItem asChild>
        <button disabled={disabled} type="button">
          <HStack gap="2" minW="0" w="full">
            <Icon as={ImagePlusIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
            <Text flex="1" fontSize="xs">
              {t('widgets.canvas.contextMenu.createFromBbox')}
            </Text>
            <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
          </HStack>
        </button>
      </Menu.TriggerItem>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="13rem" py="1">
            {CREATE_FROM_BBOX_ITEMS.map((item) => (
              <CreateFromBboxMenuItem
                key={item.destination}
                destination={item.destination}
                disabled={disabled}
                labelKey={item.labelKey}
                onCreate={onCreate}
              />
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

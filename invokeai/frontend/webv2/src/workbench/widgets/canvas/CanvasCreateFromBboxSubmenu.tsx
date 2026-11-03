import type { CreateFromBboxDestination } from '@workbench/canvas-operations/api';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
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

/**
 * What a frame snapshot can become, in the order every host lists them: the canvas context menu's Create from frame
 * submenu and the header's Frame menu.
 */
export const CreateFromBboxMenuItems = ({
  disabled,
  onCreate,
  unavailableReason,
}: {
  disabled: boolean;
  onCreate: (destination: CreateFromBboxDestination) => void;
  unavailableReason?: string;
}) => {
  const { t } = useTranslation();

  return CREATE_FROM_BBOX_ITEMS.map((item) => (
    <CreateFromBboxMenuItem
      key={item.destination}
      destination={item.destination}
      disabled={disabled}
      label={t(item.labelKey)}
      unavailableReason={unavailableReason}
      onCreate={onCreate}
    />
  ));
};

const CreateFromBboxMenuItem = ({
  destination,
  onCreate,
  ...item
}: {
  destination: CreateFromBboxDestination;
  disabled: boolean;
  label: string;
  onCreate: (destination: CreateFromBboxDestination) => void;
  unavailableReason: string | undefined;
}) => {
  const onSelect = useCallback(() => onCreate(destination), [destination, onCreate]);

  return <MenuActionItem {...item} icon={ImagePlusIcon} value={destination} onSelect={onSelect} />;
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
            <Text flex="1" fontSize="md">
              {t('widgets.canvas.contextMenu.createFromBbox')}
            </Text>
            <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
          </HStack>
        </button>
      </Menu.TriggerItem>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="13rem" py="1">
            <CreateFromBboxMenuItems disabled={disabled} onCreate={onCreate} />
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

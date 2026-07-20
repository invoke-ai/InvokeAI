import type { CanvasGallerySaveRegion } from '@workbench/canvas-operations/api';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { MenuContent } from '@platform/ui';
import { ChevronRightIcon, SaveIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const SUBMENU_POSITIONING = { placement: 'right-start' } as const;

export const CanvasSaveToGallerySubmenu = ({
  disabled,
  onSave,
}: {
  disabled: boolean;
  onSave: (region: CanvasGallerySaveRegion) => void;
}) => {
  const { t } = useTranslation();
  const onSaveCanvas = useCallback(() => onSave('canvas'), [onSave]);
  const onSaveBbox = useCallback(() => onSave('bbox'), [onSave]);

  return (
    <Menu.Root positioning={SUBMENU_POSITIONING}>
      <Menu.TriggerItem asChild>
        <button disabled={disabled} type="button">
          <HStack gap="2" minW="0" w="full">
            <Icon as={SaveIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
            <Text flex="1" fontSize="xs">
              {t('widgets.canvas.contextMenu.saveToGallery')}
            </Text>
            <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
          </HStack>
        </button>
      </Menu.TriggerItem>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="13rem" py="1">
            <Menu.Item disabled={disabled} value="save-canvas" onSelect={onSaveCanvas}>
              <HStack gap="2" minW="0" w="full">
                <Icon as={SaveIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
                <Text flex="1" fontSize="xs">
                  {t('widgets.canvas.contextMenu.saveCanvasToGallery')}
                </Text>
              </HStack>
            </Menu.Item>
            <Menu.Item disabled={disabled} value="save-bbox" onSelect={onSaveBbox}>
              <HStack gap="2" minW="0" w="full">
                <Icon as={SaveIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
                <Text flex="1" fontSize="xs">
                  {t('widgets.canvas.contextMenu.saveBboxToGallery')}
                </Text>
              </HStack>
            </Menu.Item>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

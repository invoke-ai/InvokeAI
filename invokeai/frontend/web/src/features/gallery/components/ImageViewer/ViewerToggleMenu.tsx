import {
  Button,
  Flex,
  Icon,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Text,
} from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCheckBold, PiEyeBold, PiPencilBold } from 'react-icons/pi';

import { useImageViewer } from './useImageViewer';

export const ViewerToggleMenu = () => {
  const { t } = useTranslation();
  const { isOpen, onClose, onOpen } = useImageViewer();

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Button variant="outline" data-testid="toggle-viewer-menu-button">
          <Flex gap={3} w="full" alignItems="center">
            {isOpen ? <Icon as={PiEyeBold} /> : <Icon as={PiPencilBold} />}
            <Text fontSize="md">{isOpen ? t('common.viewing') : t('common.editing')}</Text>
            <Icon as={PiCaretDownBold} />
          </Flex>
        </Button>
      </PopoverTrigger>
      <PopoverContent p={2}>
        <PopoverArrow />
        <PopoverBody>
          <Flex flexDir="column">
            <Button onClick={onOpen} variant="ghost" h="auto" w="auto" p={2}>
              <Flex gap={2} w="full">
                <Icon as={PiCheckBold} visibility={isOpen ? 'visible' : 'hidden'} />
                <Flex flexDir="column" gap={2} alignItems="flex-start">
                  <Text fontWeight="semibold" color="base.100">
                    {t('common.viewing')}
                  </Text>
                  <Text fontWeight="normal" color="base.300">
                    {t('common.viewingDesc')}
                  </Text>
                </Flex>
              </Flex>
            </Button>
            <Button onClick={onClose} variant="ghost" h="auto" w="auto" p={2}>
              <Flex gap={2} w="full">
                <Icon as={PiCheckBold} visibility={isOpen ? 'hidden' : 'visible'} />
                <Flex flexDir="column" gap={2} alignItems="flex-start">
                  <Text fontWeight="semibold" color="base.100">
                    {t('common.editing')}
                  </Text>
                  <Text fontWeight="normal" color="base.300">
                    {t('common.editingDesc')}
                  </Text>
                </Flex>
              </Flex>
            </Button>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

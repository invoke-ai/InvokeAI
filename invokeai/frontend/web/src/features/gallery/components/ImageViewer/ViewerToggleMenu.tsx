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
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCheckBold, PiEyeBold, PiPencilBold } from 'react-icons/pi';

export const ViewerToggleMenu = () => {
  const { t } = useTranslation();
  const imageViewer = useImageViewer();
  useHotkeys('z', imageViewer.onToggle, [imageViewer]);
  useHotkeys('esc', imageViewer.onClose, [imageViewer]);

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Button variant="outline" data-testid="toggle-viewer-menu-button" pointerEvents="auto">
          <Flex gap={3} w="full" alignItems="center">
            {imageViewer.isOpen ? <Icon as={PiEyeBold} /> : <Icon as={PiPencilBold} />}
            <Text fontSize="md">{imageViewer.isOpen ? t('common.viewing') : t('common.editing')}</Text>
            <Icon as={PiCaretDownBold} />
          </Flex>
        </Button>
      </PopoverTrigger>
      <PopoverContent p={2} pointerEvents="auto">
        <PopoverArrow />
        <PopoverBody>
          <Flex flexDir="column">
            <Button onClick={imageViewer.onOpen} variant="ghost" h="auto" w="auto" p={2}>
              <Flex gap={2} w="full">
                <Icon as={PiCheckBold} visibility={imageViewer.isOpen ? 'visible' : 'hidden'} />
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
            <Button onClick={imageViewer.onClose} variant="ghost" h="auto" w="auto" p={2}>
              <Flex gap={2} w="full">
                <Icon as={PiCheckBold} visibility={imageViewer.isOpen ? 'hidden' : 'visible'} />
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

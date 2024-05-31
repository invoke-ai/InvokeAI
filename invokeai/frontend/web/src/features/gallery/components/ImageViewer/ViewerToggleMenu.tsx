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
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCheckBold, PiEyeBold, PiImagesBold, PiPencilBold } from 'react-icons/pi';

import { useImageViewer } from './useImageViewer';

export const ViewerToggleMenu = () => {
  const { t } = useTranslation();
  const { viewerMode, openEditor, openViewer, openCompare } = useImageViewer();
  const icon = useMemo(() => {
    if (viewerMode === 'view') {
      return <Icon as={PiEyeBold} />;
    }
    if (viewerMode === 'edit') {
      return <Icon as={PiPencilBold} />;
    }
    if (viewerMode === 'compare') {
      return <Icon as={PiImagesBold} />;
    }
  }, [viewerMode]);
  const label = useMemo(() => {
    if (viewerMode === 'view') {
      return t('common.viewing');
    }
    if (viewerMode === 'edit') {
      return t('common.editing');
    }
    if (viewerMode === 'compare') {
      return t('common.comparing');
    }
  }, [t, viewerMode]);

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Button variant="outline" data-testid="toggle-viewer-menu-button">
          <Flex gap={3} w="full" alignItems="center">
            {icon}
            <Text fontSize="md">{label}</Text>
            <Icon as={PiCaretDownBold} />
          </Flex>
        </Button>
      </PopoverTrigger>
      <PopoverContent p={2}>
        <PopoverArrow />
        <PopoverBody>
          <Flex flexDir="column">
            <Button onClick={openViewer} variant="ghost" h="auto" w="auto" p={2}>
              <Flex gap={2} w="full">
                <Icon as={PiCheckBold} visibility={viewerMode === 'view' ? 'visible' : 'hidden'} />
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
            <Button onClick={openEditor} variant="ghost" h="auto" w="auto" p={2}>
              <Flex gap={2} w="full">
                <Icon as={PiCheckBold} visibility={viewerMode === 'edit' ? 'visible' : 'hidden'} />
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
            <Button onClick={openCompare} variant="ghost" h="auto" w="auto" p={2}>
              <Flex gap={2} w="full">
                <Icon as={PiCheckBold} visibility={viewerMode === 'compare' ? 'visible' : 'hidden'} />
                <Flex flexDir="column" gap={2} alignItems="flex-start">
                  <Text fontWeight="semibold" color="base.100">
                    {t('common.comparing')}
                  </Text>
                  <Text fontWeight="normal" color="base.300">
                    {t('common.comparingDesc')}
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

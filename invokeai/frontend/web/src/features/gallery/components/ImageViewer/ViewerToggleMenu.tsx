import { ButtonGroup, Flex, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { memo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiPencilBold } from 'react-icons/pi';

export const ViewerToggle = memo(() => {
  const { t } = useTranslation();
  const imageViewer = useImageViewer();
  useHotkeys('z', imageViewer.onToggle, [imageViewer]);
  useHotkeys('esc', imageViewer.onClose, [imageViewer]);

  return (
    <Flex gap={4} alignItems="center" justifyContent="center">
      <ButtonGroup size="md">
        <Tooltip
          hasArrow
          label={
            <Flex flexDir="column">
              <Text fontWeight="semibold">{t('common.viewing')}</Text>
              <Text fontWeight="normal">{t('common.viewingDesc')}</Text>
            </Flex>
          }
        >
          <IconButton
            icon={<PiEyeBold />}
            onClick={imageViewer.onOpen}
            variant={imageViewer.isOpen ? 'solid' : 'outline'}
            colorScheme={imageViewer.isOpen ? 'invokeBlue' : 'base'}
            aria-label={t('common.viewing')}
            w={12}
          />
        </Tooltip>
        <Tooltip
          hasArrow
          label={
            <Flex flexDir="column">
              <Text fontWeight="semibold">{t('common.editing')}</Text>
              <Text fontWeight="normal">{t('common.editingDesc')}</Text>
            </Flex>
          }
        >
          <IconButton
            icon={<PiPencilBold />}
            onClick={imageViewer.onClose}
            variant={!imageViewer.isOpen ? 'solid' : 'outline'}
            colorScheme={!imageViewer.isOpen ? 'invokeBlue' : 'base'}
            aria-label={t('common.editing')}
            w={12}
          />
        </Tooltip>
      </ButtonGroup>
    </Flex>
  );
});

ViewerToggle.displayName = 'ViewerToggle';
